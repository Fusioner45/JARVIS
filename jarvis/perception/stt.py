import asyncio
import numpy as np
from faster_whisper import WhisperModel
from jarvis.utils.config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
from jarvis.utils.logger import audio_log as log, perf_tracker

class SpeechToText:
    def __init__(self):
        log.info(f"Chargement Whisper {WHISPER_MODEL_SIZE} sur {WHISPER_DEVICE}...")
        self.model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)

    async def transcribe(self, audio_frames: list[bytes], language: str = "fr") -> str:
        loop = asyncio.get_running_loop()

        def _sync_transcribe():
            audio_np = np.frombuffer(b"".join(audio_frames), dtype=np.int16).astype(np.float32) / 32768.0
            with perf_tracker(log, "Transcription Whisper"):
                segments, _ = self.model.transcribe(audio_np, language=language, beam_size=5)
                return " ".join(seg.text for seg in segments).strip()

        try:
            return await loop.run_in_executor(None, _sync_transcribe)
        except Exception as e:
            log.error(f"STT Error: {e}")
            return ""
