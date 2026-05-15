import asyncio
import numpy as np
from faster_whisper import WhisperModel
from jarvis.utils.config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
from jarvis.utils.logger import stt_log as log

class SpeechToText:
    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
    ):
        log.info(f"Chargement de Whisper ({model_size}) sur {device}...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        log.info("✅ Modèle Whisper prêt.")

    async def transcribe(self, audio_frames: list[bytes], language: str = "fr") -> str:
        """Transcrit le buffer audio en texte via un exécuteur séparé."""
        loop = asyncio.get_running_loop()

        def _sync_transcribe():
            audio_np = np.frombuffer(b"".join(audio_frames), dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = self.model.transcribe(
                audio_np,
                language=language,
                task="transcribe",
                initial_prompt="Conversation en français. L'utilisateur parle de ses projets.",
                beam_size=5,
                best_of=5,
                suppress_tokens=[-1],
            )
            return " ".join(seg.text for seg in segments).strip()

        try:
            return await loop.run_in_executor(None, _sync_transcribe)
        except Exception as e:
            log.error(f"Erreur STT : {e}")
            if "CUDA out of memory" in str(e):
                import torch
                torch.cuda.empty_cache()
            return ""
