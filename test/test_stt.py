import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import scipy.io.wavfile as wav

model = WhisperModel("medium", device="cuda", compute_type="float16")

duration = 5
samplerate = 16000

print("Parle pendant 5 secondes...")

audio = sd.rec(int(duration * samplerate),
               samplerate=samplerate,
               channels=1,
               dtype='float32')

sd.wait()

audio = audio.flatten()

segments, info = model.transcribe(audio, language="fr")

text = " ".join([seg.text for seg in segments])

print("Transcription:")
print(text)