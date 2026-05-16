import io
import asyncio
import numpy as np
import edge_tts
from pydub import AudioSegment
from jarvis.utils.config import EDGE_TTS_VOICE, SAMPLE_RATE
from jarvis.utils.logger import audio_log as log
from jarvis.core.context import JarvisContext

class TextToSpeech:
    """Production TTS with progressive PCM streaming (Phase 8)."""

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
            mp3_buffer = bytearray()
            loop = asyncio.get_running_loop()

            async for chunk in communicate.stream():
                if context.stop_event.is_set():
                    log.debug("TTS stream aborted.")
                    return
                if chunk["type"] == "audio":
                    mp3_buffer.extend(chunk["data"])

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
                chunk_size = SAMPLE_RATE * 2
                for i in range(0, len(pcm), chunk_size):
                    if context.stop_event.is_set(): break
                    await context.audio_output_queue.put(pcm[i:i+chunk_size])

        except asyncio.CancelledError: pass
        except Exception as e:
            log.error(f"TTS Runtime Error: {e}")
