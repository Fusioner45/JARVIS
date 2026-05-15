import io
import asyncio
import numpy as np
import edge_tts
from pydub import AudioSegment
from jarvis.utils.config import EDGE_TTS_VOICE, SAMPLE_RATE, FRAME_SIZE
from jarvis.utils.logger import audio_log as log, perf_tracker
from jarvis.core.context import JarvisContext

class TextToSpeech:
    def __init__(self):
        self.voice = EDGE_TTS_VOICE

    async def speak(self, text: str, context: JarvisContext):
        """Generates speech and pushes PCM data to context queue. Respects context.stop_event."""
        if not text or context.stop_event.is_set():
            return

        log.info(f"🎙️ TTS : {text[:50]}...")
        context.is_speaking = True

        try:
            # edge-tts communicate supports cancellation via asyncio.Task
            communicate = edge_tts.Communicate(text, self.voice)
            mp3_data = io.BytesIO()

            async for chunk in communicate.stream():
                if context.stop_event.is_set():
                    log.debug("TTS Stream interrupted by stop_event")
                    return
                if chunk["type"] == "audio":
                    mp3_data.write(chunk["data"])

            if mp3_data.tell() == 0:
                return

            mp3_data.seek(0)

            # Offload decoding to thread to prevent event loop blocking
            loop = asyncio.get_running_loop()
            def _decode():
                try:
                    # ffmpeg/pydub decoding is heavy
                    audio = AudioSegment.from_file(mp3_data, format="mp3")
                    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
                    return np.frombuffer(audio.raw_data, dtype=np.int16)
                except Exception as e:
                    log.error(f"Erreur décodage MP3 : {e}")
                    return None

            pcm = await loop.run_in_executor(None, _decode)

            if pcm is not None and not context.stop_event.is_set():
                await context.audio_output_queue.put(pcm)

        except asyncio.CancelledError:
            log.debug("Tâche TTS annulée.")
        except Exception as e:
            log.error(f"TTS Error: {e}")
        finally:
            # Note: is_speaking should only be set to False when the playback is actually DONE
            # which is handled by the playback_manager in orchestrator
            pass
