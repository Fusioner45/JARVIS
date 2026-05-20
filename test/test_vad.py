import sounddevice as sd
import numpy as np
from jarvis.perception.audio import VoiceActivityDetector

vad = VoiceActivityDetector()

def callback(indata, frames, time, status):
    audio = (indata[:, 0] * 32767).astype(np.int16).tobytes()

    speaking = vad.is_speech(audio)

    print("VOICE" if speaking else "silence")

with sd.InputStream(callback=callback, channels=1, samplerate=16000, blocksize=512):
    print("Parle...")
    input()