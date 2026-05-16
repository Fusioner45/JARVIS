import io
import asyncio
import subprocess
import numpy as np
import edge_tts
from jarvis.utils.config import EDGE_TTS_VOICE, SAMPLE_RATE
from jarvis.utils.logger import audio_log as log
from jarvis.core.context import JarvisContext

# ffmpeg: décode MP3 streamed → PCM s16le mono
_FFMPEG_CMD = [
    "ffmpeg", "-loglevel", "quiet",
    "-f", "mp3", "-i", "pipe:0",
    "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1"
]
# Taille d'un chunk PCM = 100ms à 16kHz en s16le (2 bytes/sample)
_PCM_CHUNK_BYTES = 16000 // 10 * 2  # 3200 bytes


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=2)
        return True
    except Exception:
        return False


class TextToSpeech:
    """TTS avec streaming PCM via pipe ffmpeg (Phase 9 — faible latence)."""

    def __init__(self):
        self.voice = EDGE_TTS_VOICE
        self._use_ffmpeg = _ffmpeg_available()
        if self._use_ffmpeg:
            log.info("✅ TTS: mode streaming ffmpeg actif (faible latence).")
        else:
            log.warning("⚠️ TTS: ffmpeg absent. Fallback pydub (latence accrue).")

    async def speak(self, text: str, context: JarvisContext):
        if not text or context.stop_event.is_set():
            return

        log.info(f"🎙️ TTS: {text[:50]}...")
        context.is_speaking = True

        try:
            if self._use_ffmpeg:
                await self._speak_ffmpeg(text, context)
            else:
                await self._speak_pydub(text, context)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"TTS Runtime Error: {e}")

    async def _speak_ffmpeg(self, text: str, context: JarvisContext):
        """Streaming MP3 → ffmpeg → chunks PCM. Latence ~150–300ms."""
        communicate = edge_tts.Communicate(text, self.voice)

        proc = await asyncio.create_subprocess_exec(
            *_FFMPEG_CMD,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

        async def _feed_mp3():
            try:
                async for chunk in communicate.stream():
                    if context.stop_event.is_set():
                        break
                    if chunk["type"] == "audio":
                        proc.stdin.write(chunk["data"])
                        await proc.stdin.drain()
            except Exception as e:
                log.error(f"TTS feed error: {e}")
            finally:
                try:
                    proc.stdin.close()
                except Exception:
                    pass

        feed_task = asyncio.create_task(_feed_mp3())

        try:
            while True:
                if context.stop_event.is_set():
                    proc.kill()
                    break

                raw = await proc.stdout.read(_PCM_CHUNK_BYTES)
                if not raw:
                    break

                pcm = np.frombuffer(raw, dtype=np.int16)

                # Resample si SAMPLE_RATE != 16000
                if SAMPLE_RATE != 16000:
                    factor = SAMPLE_RATE / 16000
                    idx = np.clip(
                        (np.arange(int(len(pcm) * factor)) / factor).astype(int),
                        0, len(pcm) - 1
                    )
                    pcm = pcm[idx]

                if not context.stop_event.is_set():
                    await context.audio_output_queue.put(pcm)
        finally:
            feed_task.cancel()
            try:
                await proc.wait()
            except Exception:
                pass

    async def _speak_pydub(self, text: str, context: JarvisContext):
        """Fallback: accumule le MP3 complet avant décodage."""
        from pydub import AudioSegment

        communicate = edge_tts.Communicate(text, self.voice)
        mp3_buffer = bytearray()

        async for chunk in communicate.stream():
            if context.stop_event.is_set():
                return
            if chunk["type"] == "audio":
                mp3_buffer.extend(chunk["data"])

        if not mp3_buffer:
            return

        loop = asyncio.get_running_loop()

        def _decode(data: bytes) -> np.ndarray:
            seg = AudioSegment.from_file(io.BytesIO(data), format="mp3")
            seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
            return np.frombuffer(seg.raw_data, dtype=np.int16)

        pcm = await loop.run_in_executor(None, _decode, bytes(mp3_buffer))

        if pcm is not None:
            chunk_size = SAMPLE_RATE * 2  # 2s par chunk
            for i in range(0, len(pcm), chunk_size):
                if context.stop_event.is_set():
                    break
                await context.audio_output_queue.put(pcm[i:i + chunk_size])
