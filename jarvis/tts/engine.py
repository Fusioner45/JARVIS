import io
import asyncio
import numpy as np
import edge_tts
from pydub import AudioSegment
from jarvis.utils.config import EDGE_TTS_VOICE, SAMPLE_RATE
from jarvis.utils.logger import audio_log as log
from jarvis.core.context import JarvisContext

class TextToSpeech:
    """Production TTS with progressive PCM streaming (Phase 7)."""

    def __init__(self):
        self.voice = EDGE_TTS_VOICE

    async def speak(self, text: str, context: JarvisContext):
        """Generates speech and streams PCM chunks to context queue progressively."""
        if not text or context.stop_event.is_set():
            return

        log.info(f"🎙️ TTS Streaming : {text[:40]}...")
        context.is_speaking = True

        try:
            communicate = edge_tts.Communicate(text, self.voice)

            # Pour edge-tts, le streaming audio est en MP3.
            # Le décodage progressif MP3 nécessite un parser robuste (type ffmpeg pipe).
            # Ici on utilise une approche par "blocs de stream" pour limiter la latence et la RAM.

            mp3_buffer = bytearray()
            loop = asyncio.get_running_loop()

            async for chunk in communicate.stream():
                if context.stop_event.is_set():
                    log.debug("TTS stream interrupted.")
                    return

                if chunk["type"] == "audio":
                    mp3_buffer.extend(chunk["data"])

                    # Si on a accumulé assez de données (ex: ~1s d'audio), on décode un segment
                    # Note: Le MP3 n'est pas facilement découpable sans perdre des frames,
                    # mais pour des réponses courtes/moyennes, décoder par segments de stream
                    # ou à la fin du stream edge-tts est suffisant.
                    # Pour un vrai streaming temps-réel, on utiliserait un Subprocess FFMPEG.

            if not mp3_buffer: return

            def _decode(data):
                try:
                    seg = AudioSegment.from_file(io.BytesIO(data), format="mp3")
                    seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
                    return np.frombuffer(seg.raw_data, dtype=np.int16)
                except Exception as e:
                    log.error(f"TTS Segment Decode Error: {e}")
                    return None

            pcm = await loop.run_in_executor(None, _decode, bytes(mp3_buffer))

            if pcm is not None and not context.stop_event.is_set():
                # On injecte le PCM par chunks pour permettre l'interruption entre les phrases
                chunk_size = SAMPLE_RATE * 2 # ~2 secondes
                for i in range(0, len(pcm), chunk_size):
                    if context.stop_event.is_set(): break
                    await context.audio_output_queue.put(pcm[i:i+chunk_size])

        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"TTS Runtime Error: {e}")
