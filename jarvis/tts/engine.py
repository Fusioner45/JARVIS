import os
import asyncio
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
from kokoro_onnx import Kokoro
from jarvis.utils.config import KOKORO_VOICE, KOKORO_MODEL_PATH, KOKORO_VOICES_PATH, SAMPLE_RATE
from jarvis.utils.logger import audio_log as log
from jarvis.core.context import JarvisContext

class TextToSpeech:
    """Kokoro TTS Engine (Phase 10 - 100% Local & Low Latency)."""

    def __init__(self):
        log.info("--- INITIALISATION KOKORO TTS ---")
        self.available = False
        self.kokoro = None

        if not self._validate_assets():
            log.error("❌ Les assets Kokoro sont invalides ou manquants. TTS désactivé.")
            return

        try:
            # Convert to absolute paths (prevents Windows cp1252 encoding errors during model load)
            model_abs = os.path.abspath(KOKORO_MODEL_PATH)
            voices_abs = os.path.abspath(KOKORO_VOICES_PATH)
            # Automatic Device Detection (onnxruntime-gpu handles this if installed)
            self.kokoro = Kokoro(model_abs, voices_abs)
            log.info("✅ Kokoro TTS: Modèle chargé avec succès.")
            self.available = True
        except Exception as e:
            log.critical(f"❌ Erreur critique lors du chargement Kokoro : {e}")
            if "INVALID_PROTOBUF" in str(e):
                log.error("🚨 Le modèle ONNX semble corrompu (Protobuf error).")
                self._handle_corruption(KOKORO_MODEL_PATH)
            self.available = False

    def _validate_assets(self) -> bool:
        """Checks size and content of model files to detect corruption/HTML/LFS."""
        for path in [KOKORO_MODEL_PATH, KOKORO_VOICES_PATH]:
            if not os.path.exists(path):
                log.warning(f"⚠️ Asset manquant : {path}")
                return False

            size_mb = os.path.getsize(path) / (1024 * 1024)
            log.debug(f"🔍 Asset {os.path.basename(path)} : {size_mb:.2f} MB")

            # Kokoro ONNX is ~300MB+, Voices is ~15MB+
            # 70MB as reported in issue is definitely truncated or a pointer.
            min_size = 250 if path == KOKORO_MODEL_PATH else 10
            if size_mb < min_size:
                log.error(f"❌ Asset trop petit ({size_mb:.2f}MB < {min_size}MB) : {path}")
                self._handle_corruption(path)
                return False

            # Content Check: Detect HTML or LFS
            try:
                with open(path, 'rb') as f:
                    head = f.read(100).decode('utf-8', errors='ignore')
                    if "<!DOCTYPE html>" in head or "<html" in head:
                        log.error(f"❌ Fichier HTML détecté au lieu du modèle : {path}")
                        self._handle_corruption(path)
                        return False
                    if "version https://git-lfs.github.com/spec" in head:
                        log.error(f"❌ Pointeur Git LFS détecté au lieu du modèle : {path}")
                        self._handle_corruption(path)
                        return False
            except Exception as e:
                log.error(f"Erreur lecture validation asset : {e}")
                return False

        return True

    def _handle_corruption(self, path: str):
        """Removes corrupt file and logs instructions."""
        try:
            log.warning(f"🗑️ Suppression du fichier corrompu : {path}")
            os.remove(path)
            log.info("💡 Relancez setup_jarvis.bat pour retélécharger proprement le modèle.")
        except Exception as e:
            log.error(f"Impossible de supprimer {path} : {e}")

    async def speak(self, text: str, context: JarvisContext):
        if not text or context.stop_event.is_set() or not self.available or not self.kokoro:
            return

        log.info(f"🎙️ Kokoro TTS: {text[:50]}...")
        context.is_speaking = True

        try:
            loop = asyncio.get_running_loop()
            samples, sr = await loop.run_in_executor(None, self._generate, text)

            if samples is None or len(samples) == 0:
                log.warning("⚠️ Kokoro n'a généré aucun audio.")
                return

            if sr != SAMPLE_RATE:
                samples = self._resample(samples, sr, SAMPLE_RATE)

            pcm = (samples * 32767).astype(np.int16)

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
        try:
            return self.kokoro.create(text, voice=KOKORO_VOICE, speed=1.0, lang="en-us")
        except Exception as e:
            log.error(f"Kokoro Generation Error: {e}")
            return None, None

    def _resample(self, audio, from_sr, to_sr):
        if from_sr == to_sr:
            return audio
        target_len = int(len(audio) * to_sr / from_sr)
        indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
