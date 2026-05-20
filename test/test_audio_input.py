import sounddevice as sd
import numpy as np

def callback(indata, frames, time, status):
    volume = np.linalg.norm(indata) * 10
    print(f"Volume: {volume:.4f}")

with sd.InputStream(callback=callback, channels=1, samplerate=16000):
    print("Parle dans le micro...")
    input()