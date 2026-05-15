import asyncio
import numpy as np
import sounddevice as sd
import torch
import os
from jarvis.utils.config import SAMPLE_RATE, FRAME_SIZE
from jarvis.utils.logger import audio_log as log
from jarvis.core.context import JarvisContext

async def audio_frame_generator(context: JarvisContext):
    """Captures microphone audio and yields PCM frames. Discards if JARVIS is speaking."""
    loop = asyncio.get_running_loop()
    q = asyncio.Queue(maxsize=200) # Buffer de sécurité de ~6 secondes

    def callback(indata, frames, time, status):
        if status: log.warning(f"Audio Input Status: {status}")
        if context.is_speaking: return

        # Si la queue est pleine, on vide le plus vieux pour rester temps réel
        if q.full():
            try: loop.call_soon_threadsafe(q.get_nowait)
            except: pass

        mono = indata[:, 0] if indata.ndim > 1 else indata
        pcm16 = (mono * 32767).astype(np.int16).tobytes()
        loop.call_soon_threadsafe(q.put_nowait, pcm16)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=FRAME_SIZE, callback=callback):
            log.info("🎤 Microphone actif - Flux de perception ouvert.")
            while True:
                frame = await q.get()
                yield frame
    except Exception as e:
        log.critical(f"Erreur fatale microphone : {e}")

class VoiceActivityDetector:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        # Force CPU pour le VAD afin de libérer de la VRAM pour Whisper/LLM
        torch.set_num_threads(1)
        # Check if local model exists or download
        self.model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', onnx=True, force_reload=False)
        self.sample_rate = sample_rate

    def is_speech(self, frame: bytes, threshold: float = 0.5) -> bool:
        if not frame: return False

        audio_int16 = np.frombuffer(frame, dtype=np.int16)

        # Filtre d'énergie adaptatif (Noise Gate)
        # On calcule le RMS pour ignorer le bruit blanc électrique avant le VAD
        rms = np.sqrt(np.mean(audio_int16.astype(np.float32)**2)) / 32768.0
        if rms < 0.005: return False # Seuil très bas pour capter les chuchotements

        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        try:
            with torch.no_grad():
                # Silero VAD attend un tensor de forme [1, samples]
                input_tensor = torch.from_numpy(audio_float32).unsqueeze(0)
                confidence = self.model(input_tensor, self.sample_rate).item()
            return confidence > threshold
        except Exception as e:
            log.error(f"VAD Inference Error: {e}")
            return False
