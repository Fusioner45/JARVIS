import asyncio
import numpy as np
import sounddevice as sd
import torch
from jarvis.utils.config import SAMPLE_RATE, FRAME_SIZE
from jarvis.utils.logger import audio_log as log
from jarvis.core.context import JarvisContext

async def audio_frame_generator(context: JarvisContext):
    """
    Capture audio depuis le microphone et yield des frames PCM16.
    Ignore la capture si context.is_speaking est True (Anti-Larsen).
    """
    loop = asyncio.get_running_loop()
    q = asyncio.Queue()

    def callback(indata, frames, time, status):
        if status:
            log.warning(f"Statut stream audio : {status}")

        # Anti-Larsen : On ne met rien dans la queue si JARVIS parle
        if context.is_speaking:
            return

        mono = indata[:, 0] if indata.ndim > 1 else indata
        pcm16 = (mono * 32767).astype(np.int16).tobytes()
        loop.call_soon_threadsafe(q.put_nowait, pcm16)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SIZE,
        callback=callback,
    ):
        log.info("🎤 Microphone ouvert - Traitement temps réel actif.")
        while True:
            frame = await q.get()
            if frame is None:
                break
            yield frame

class VoiceActivityDetector:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        log.info("Initialisation de Silero VAD (ONNX)...")
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=True
        )
        self.sample_rate = sample_rate

    def is_speech(self, frame: bytes, threshold: float = 0.5, proximity_threshold: float = 0.01) -> bool:
        audio_int16 = np.frombuffer(frame, dtype=np.int16)

        # Filtre d'énergie (RMS) pour ignorer les bruits lointains
        rms = np.sqrt(np.mean(audio_int16.astype(np.float32)**2)) / 32768.0
        if rms < proximity_threshold:
            return False

        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        tensor_input = torch.from_numpy(audio_float32).unsqueeze(0)

        with torch.no_grad():
            confidence = self.model(tensor_input, self.sample_rate).item()
        return confidence > threshold
