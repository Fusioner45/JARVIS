import numpy as np

def test_agc():
    AGC_TARGET_RMS = 0.10
    AGC_MAX_GAIN = 8.0
    smoothed_gain = 1.0

    # Simulate low volume audio
    mono = np.random.randn(512).astype(np.float32) * 0.01
    rms = np.sqrt(np.mean(mono ** 2))
    print(f"Initial RMS: {rms:.4f}")

    for i in range(100):
        rms = np.sqrt(np.mean(mono ** 2))
        if rms > 0.001:
            target_gain = AGC_TARGET_RMS / rms
            target_gain = min(target_gain, AGC_MAX_GAIN)
            smoothed_gain = 0.95 * smoothed_gain + 0.05 * target_gain
        else:
            smoothed_gain = 0.99 * smoothed_gain + 0.01 * 1.0

    print(f"Final Smoothed Gain: {smoothed_gain:.4f}")
    mono_boosted = mono * smoothed_gain
    final_rms = np.sqrt(np.mean(mono_boosted ** 2))
    print(f"Final RMS: {final_rms:.4f}")

    # Test clipping
    mono_loud = np.random.randn(512).astype(np.float32) * 2.0
    mono_clipped = np.clip(mono_loud, -1.0, 1.0)
    print(f"Max after clipping: {np.max(mono_clipped)}, Min: {np.min(mono_clipped)}")

test_agc()
