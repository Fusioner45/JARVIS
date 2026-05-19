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
_PCM_CHUNK_BYTES = 16000 // 10 * 2


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
            # First Attempt: Edge-TTS
            success = await self._try_edge_tts(text, context)

            # Second Attempt: Piper (TODO)
            if not success and not context.stop_event.is_set():
                success = await self._speak_piper(text, context)

            # Final Attempt: Fallback Google TTS (via pydub/io)
            if not success and not context.stop_event.is_set():
                log.warning("🔄 Edge-TTS a échoué. Tentative de fallback gTTS...")
                await self._speak_gtts(text, context)

            if not success:
                log.error("❌ Tous les services TTS ont échoué.")
                context.is_speaking = False

        except asyncio.CancelledError:
            context.is_speaking = False
        except Exception as e:
            log.error(f"TTS Runtime Error: {e}")
            context.is_speaking = False
        finally:
            # Note: is_speaking is also reset in orchestrator's playback_manager
            # when audio_output_queue is empty.
            pass

    async def _try_edge_tts(self, text: str, context: JarvisContext) -> bool:
        try:
            if self._use_ffmpeg:
                return await self._speak_ffmpeg(text, context)
            else:
                return await self._speak_pydub(text, context)
        except Exception as e:
            log.error(f"Edge-TTS Attempt Failed: {e}")
            return False

    async def _speak_ffmpeg(self, text: str, context: JarvisContext) -> bool:
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
                        proc.kill()
                        break
                    if chunk["type"] == "audio":
                        proc.stdin.write(chunk["data"])
                        await proc.stdin.drain()
            except Exception:
                proc.kill()
            finally:
                try: proc.stdin.close()
                except: pass

        feed_task = asyncio.create_task(_feed_mp3())
        has_audio = False

        try:
            while True:
                if context.stop_event.is_set():
                    break
                raw = await proc.stdout.read(_PCM_CHUNK_BYTES)
                if not raw: break
                has_audio = True
                pcm = np.frombuffer(raw, dtype=np.int16)
                if SAMPLE_RATE != 16000:
                    factor = SAMPLE_RATE / 16000
                    idx = np.clip((np.arange(int(len(pcm) * factor)) / factor).astype(int), 0, len(pcm) - 1)
                    pcm = pcm[idx]
                await context.audio_output_queue.put(pcm)
            await proc.wait()
            return has_audio
        finally:
            feed_task.cancel()

    async def _speak_pydub(self, text: str, context: JarvisContext) -> bool:
        from pydub import AudioSegment
        communicate = edge_tts.Communicate(text, self.voice)
        mp3_buffer = bytearray()
        async for chunk in communicate.stream():
            if context.stop_event.is_set():
                return False
            if chunk["type"] == "audio":
                mp3_buffer.extend(chunk["data"])
        if not mp3_buffer: return False

        loop = asyncio.get_running_loop()
        def _decode(data):
            try:
                seg = AudioSegment.from_file(io.BytesIO(data), format="mp3")
                seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
                return np.frombuffer(seg.raw_data, dtype=np.int16)
            except: return None

        pcm = await loop.run_in_executor(None, _decode, bytes(mp3_buffer))
        if pcm is not None:
            chunk_size = SAMPLE_RATE * 2
            for i in range(0, len(pcm), chunk_size):
                if context.stop_event.is_set():
                    return True
                await context.audio_output_queue.put(pcm[i:i + chunk_size])
            return True
        return False

    async def _speak_piper(self, text: str, context: JarvisContext) -> bool:
        """TODO: Implémenter Piper TTS pour une génération 100% locale."""
        return False

    async def _speak_gtts(self, text: str, context: JarvisContext) -> bool:
        """Robust fallback using gTTS."""
        if context.stop_event.is_set():
            return False

        try:
            from gtts import gTTS
            from pydub import AudioSegment
            tts = gTTS(text=text, lang='fr')
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)

            if context.stop_event.is_set():
                return False

            loop = asyncio.get_running_loop()
            def _decode():
                audio = AudioSegment.from_file(mp3_fp, format="mp3")
                audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
                return np.frombuffer(audio.raw_data, dtype=np.int16)

            pcm = await loop.run_in_executor(None, _decode)
            if pcm is not None:
                if context.stop_event.is_set():
                    return True
                await context.audio_output_queue.put(pcm)
                log.info("✅ Fallback gTTS réussi.")
                return True
            return False
        except Exception as e:
            log.error(f"Fallback gTTS Error: {e}")
            return False
