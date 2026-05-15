import asyncio
import numpy as np
from faster_whisper import WhisperModel
from jarvis.utils.config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
from jarvis.utils.logger import audio_log as log, perf_tracker

class SpeechToText:
    """Production-Grade STT with Hallucination Filtering (Phase 3)."""

    def __init__(self):
        log.info(f"Chargement Whisper {WHISPER_MODEL_SIZE} sur {WHISPER_DEVICE}...")
        self.model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)

        # Blacklist de phrases typiques des hallucinations de silence Whisper
        self.blacklist = {
            "Merci d'avoir regardé", "Merci d'avoir regardé cette vidéo",
            "Sous-titres réalisés par", "Mettez un pouce bleu",
            "Abonnez-vous", "J'avise que", "Transcription", "Bye", "Merci."
        }

    async def transcribe(self, audio_frames: list[bytes], language: str = "fr") -> str:
        loop = asyncio.get_running_loop()

        def _sync_transcribe():
            audio_np = np.frombuffer(b"".join(audio_frames), dtype=np.int16).astype(np.float32) / 32768.0

            with perf_tracker(log, "Whisper Transcription"):
                segments, _ = self.model.transcribe(
                    audio_np,
                    language=language,
                    beam_size=5,
                    best_of=5,
                    initial_prompt="Ceci est une conversation en français uniquement. Pas de traduction.",
                    suppress_tokens=[-1]
                )
                text = " ".join(seg.text for seg in segments).strip()

                # --- Phase 3: Hallucination Filtering ---
                if not text:
                    return ""

                # 1. Entropy/Redundancy check (too many repetitions)
                words = text.split()
                if len(words) > 10:
                    unique_ratio = len(set(words)) / len(words)
                    if unique_ratio < 0.3:
                        log.warning(f"🤫 Hallucination d'entropie détectée ({unique_ratio:.2f}) : {text}")
                        return ""

                # 2. Blacklist check
                if any(phrase.lower() in text.lower() for phrase in self.blacklist):
                    log.info(f"🤫 Hallucination de blacklist filtrée : {text}")
                    return ""

                # 3. Short duration check (Whisper tends to hallucinate 'Merci' on silence)
                duration = len(audio_np) / 16000
                if duration < 0.8 and ("merci" in text.lower() or "bye" in text.lower()):
                    log.debug(f"🤫 Hallucination courte filtrée : {text}")
                    return ""

                return text

        try:
            return await loop.run_in_executor(None, _sync_transcribe)
        except Exception as e:
            log.error(f"STT Error: {e}")
            if "CUDA out of memory" in str(e):
                import torch
                torch.cuda.empty_cache()
            return ""
