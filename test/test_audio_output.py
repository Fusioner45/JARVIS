import sounddevice as sd
import numpy as np

samplerate = 16000
duration = 2
frequency = 440

t = np.linspace(0, duration, int(samplerate * duration), False)

tone = 0.3 * np.sin(2 * np.pi * frequency * t)

sd.play(tone, samplerate)

sd.wait()

print("Son joué.")