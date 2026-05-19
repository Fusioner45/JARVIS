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
        self.available = False
        self.kokoro = None

        if not self._validate_assets():
            log.error("❌ Les assets Kokoro sont invalides ou manquants. TTS désactivé.")
            return

        try:
            # Automatic Device Detection (onnxruntime-gpu handles this if installed)
            self.kokoro = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)
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

            min_size = 250 if path == KOKORO_MODEL_PATH else 10
            if size_mb < min_size:
                log.error(f"❌ Asset trop petit ({size_mb:.2f}MB < {min_size}MB) : {path}")
                self._handle_corruption(path)
                return False

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
        if not text or context.stop_event.is_set():
            return

        log.info(f"🎙️ TTS: {text[:60]}...")
        context.is_speaking = True

        try:
            if self.available and self.kokoro:
                await self._speak_kokoro(text, context)
            else:
                # Fallback : edge-tts (Microsoft, nécessite Internet)
                await self._speak_edge_tts(text, context)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"TTS Runtime Error: {e}")
        finally:
            if context.audio_output_queue.empty():
                context.is_speaking = False

    async def _speak_kokoro(self, text: str, context: JarvisContext):
        """Kokoro local TTS (préféré)."""
        loop = asyncio.get_running_loop()
        samples, sr = await loop.run_in_executor(None, self._generate, text)

        if samples is None or len(samples) == 0:
            log.warning("⚠️ Kokoro n'a généré aucun audio. Fallback edge-tts.")
            await self._speak_edge_tts(text, context)
            return

        if sr != SAMPLE_RATE:
            samples = self._resample(samples, sr, SAMPLE_RATE)

        pcm = (samples * 32767).astype(np.int16)
        chunk_size = SAMPLE_RATE // 5
        for i in range(0, len(pcm), chunk_size):
            if context.stop_event.is_set():
                break
            await context.audio_output_queue.put(pcm[i:i + chunk_size])

    async def _speak_edge_tts(self, text: str, context: JarvisContext):
        """Fallback edge-tts (Microsoft, online)."""
        try:
            import edge_tts, io, subprocess
            import asyncio as aio

            log.info("🔄 TTS Fallback: edge-tts")
            communicate = edge_tts.Communicate(text, "fr-FR-HenriNeural")

            _FFMPEG = [
                "ffmpeg", "-loglevel", "quiet",
                "-f", "mp3", "-i", "pipe:0",
                "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1"
            ]

            try:
                proc = await aio.create_subprocess_exec(
                    *_FFMPEG,
                    stdin=aio.subprocess.PIPE,
                    stdout=aio.subprocess.PIPE,
                    stderr=aio.subprocess.DEVNULL,
                )

                async def _feed():
                    try:
                        async for chunk in communicate.stream():
                            if context.stop_event.is_set():
                                break
                            if chunk["type"] == "audio":
                                proc.stdin.write(chunk["data"])
                                await proc.stdin.drain()
                    finally:
                        try: proc.stdin.close()
                        except: pass

                feed_task = aio.create_task(_feed())
                _CHUNK = SAMPLE_RATE // 10 * 2  # 100ms PCM

                while True:
                    if context.stop_event.is_set():
                        proc.kill()
                        break
                    raw = await proc.stdout.read(_CHUNK)
                    if not raw:
                        break
                    pcm = np.frombuffer(raw, dtype=np.int16)
                    await context.audio_output_queue.put(pcm)

                feed_task.cancel()
                await proc.wait()

            except FileNotFoundError:
                # ffmpeg absent → fallback pydub
                from pydub import AudioSegment
                mp3 = bytearray()
                async for chunk in communicate.stream():
                    if context.stop_event.is_set(): return
                    if chunk["type"] == "audio": mp3.extend(chunk["data"])
                if not mp3: return
                loop = aio.get_running_loop()
                def _dec(d):
                    seg = AudioSegment.from_file(io.BytesIO(d), format="mp3")
                    seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
                    return np.frombuffer(seg.raw_data, dtype=np.int16)
                pcm = await loop.run_in_executor(None, _dec, bytes(mp3))
                if pcm is not None:
                    for i in range(0, len(pcm), SAMPLE_RATE // 5):
                        if context.stop_event.is_set(): break
                        await context.audio_output_queue.put(pcm[i:i + SAMPLE_RATE // 5])

        except Exception as e:
            log.error(f"edge-tts fallback error: {e}")

    def _generate(self, text):
        try:
            return self.kokoro.create(text, voice=KOKORO_VOICE, speed=1.0, lang="fr-fr")
        except Exception as e:
            log.error(f"Kokoro Generation Error: {e}")
            return None, None

    def _resample(self, audio, from_sr, to_sr):
        if from_sr == to_sr:
            return audio
        target_len = int(len(audio) * to_sr / from_sr)
        indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
