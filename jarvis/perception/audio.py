import asyncio
import numpy as np
import sounddevice as sd
import torch
from jarvis.utils.config import SAMPLE_RATE, FRAME_SIZE
from jarvis.utils.logger import audio_log as log
from jarvis.core.context import JarvisContext

async def audio_frame_generator(context: JarvisContext):
    """Yields audio frames from microphone. Discards input if JARVIS is speaking."""
    loop = asyncio.get_running_loop()
    q = asyncio.Queue()

    def callback(indata, frames, time, status):
        if status: log.warning(f"Audio status: {status}")
        if context.is_speaking: return

        mono = indata[:, 0] if indata.ndim > 1 else indata
        pcm16 = (mono * 32767).astype(np.int16).tobytes()
        loop.call_soon_threadsafe(q.put_nowait, pcm16)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=FRAME_SIZE, callback=callback):
        log.info("🎤 Microphone actif.")
        while True:
            frame = await q.get()
            yield frame

class VoiceActivityDetector:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', onnx=True)
        self.sample_rate = sample_rate

    def is_speech(self, frame: bytes, threshold: float = 0.5) -> bool:
        audio_int16 = np.frombuffer(frame, dtype=np.int16)
        # RMS filter
        rms = np.sqrt(np.mean(audio_int16.astype(np.float32)**2)) / 32768.0
        if rms < 0.01: return False

        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        with torch.no_grad():
            confidence = self.model(torch.from_numpy(audio_float32).unsqueeze(0), self.sample_rate).item()
        return confidence > threshold
