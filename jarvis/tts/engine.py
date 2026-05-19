import os
import asyncio
import numpy as np
from kokoro_onnx import Kokoro
from jarvis.utils.config import KOKORO_VOICE, KOKORO_MODEL_PATH, KOKORO_VOICES_PATH, SAMPLE_RATE
from jarvis.utils.logger import audio_log as log
from jarvis.core.context import JarvisContext

class TextToSpeech:
    """Kokoro TTS Engine (Phase 10 - 100% Local & Low Latency)."""

    def __init__(self):
        log.info("--- INITIALISATION KOKORO TTS ---")

        if not os.path.exists(KOKORO_MODEL_PATH) or not os.path.exists(KOKORO_VOICES_PATH):
            log.error(f"❌ Kokoro models missing: {KOKORO_MODEL_PATH} or {KOKORO_VOICES_PATH}")
            self.kokoro = None
            return

        try:
            # Automatic Device Detection (onnxruntime-gpu handles this if installed)
            self.kokoro = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)
            log.info("✅ Kokoro TTS: Modèle chargé avec succès.")
        except Exception as e:
            log.critical(f"❌ Erreur critique Kokoro : {e}")
            self.kokoro = None

    async def speak(self, text: str, context: JarvisContext):
        if not text or context.stop_event.is_set() or not self.kokoro:
            return

        log.info(f"🎙️ Kokoro TTS: {text[:50]}...")
        context.is_speaking = True

        try:
            # Kokoro creation is synchronous, offload to executor
            loop = asyncio.get_running_loop()
            samples, sr = await loop.run_in_executor(None, self._generate, text)

            if samples is None or len(samples) == 0:
                log.warning("⚠️ Kokoro n'a généré aucun audio.")
                return

            # Resample if Kokoro (24kHz) doesn't match system (16kHz)
            # Note: Kokoro-82M usually outputs at 24000Hz
            if sr != SAMPLE_RATE:
                samples = self._resample(samples, sr, SAMPLE_RATE)

            # Convert to PCM16
            pcm = (samples * 32767).astype(np.int16)

            # Injection by chunks to audio queue
            chunk_size = SAMPLE_RATE // 5 # 200ms chunks
            for i in range(0, len(pcm), chunk_size):
                if context.stop_event.is_set():
                    log.debug("TTS interruption detection.")
                    break

                chunk = pcm[i:i + chunk_size]
                await context.audio_output_queue.put(chunk)

        except asyncio.CancelledError:
            log.debug("TTS Task cancelled.")
        except Exception as e:
            log.error(f"Kokoro Runtime Error: {e}")
        finally:
            if context.audio_output_queue.empty():
                context.is_speaking = False

    def _generate(self, text):
        """Synchronous wrapper for kokoro-onnx generation."""
        try:
            # af_sky is generally a good default, or use KOKORO_VOICE from config
            return self.kokoro.create(text, voice=KOKORO_VOICE, speed=1.0, lang="en-us")
        except Exception as e:
            log.error(f"Kokoro Generation Error: {e}")
            return None, None

    def _resample(self, audio, from_sr, to_sr):
        """Simple linear interpolation resampling for mono audio."""
        if from_sr == to_sr:
            return audio
        target_len = int(len(audio) * to_sr / from_sr)
        indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
