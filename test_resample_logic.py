import numpy as np
import scipy.signal
from math import gcd

def _resample(mono, from_sr, to_sr):
    if from_sr == to_sr: return mono
    g = gcd(from_sr, to_sr)
    return scipy.signal.resample_poly(mono, to_sr // g, from_sr // g).astype(np.float32)

data = np.zeros(1600).astype(np.float32)
resampled = _resample(data, 16000, 16000)
print(f'16k to 16k: {len(resampled)}')
resampled = _resample(data, 16000, 24000)
print(f'16k to 24k: {len(resampled)}')
resampled = _resample(data, 44100, 16000)
print(f'44.1k to 16k: {len(resampled)}')
