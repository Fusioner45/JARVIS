import asyncio
import numpy as np
import sounddevice as sd
import onnxruntime as ort
import os
import time
import queue
import threading
from jarvis.utils.config import SAMPLE_RATE, FRAME_SIZE, VAD_THRESHOLD, AGC_TARGET_RMS, AGC_MAX_GAIN
from jarvis.utils.logger import audio_log as log
from jarvis.core.context import JarvisContext

# Ajout pour resampling natif → 16kHz
try:
    from scipy.signal import resample_poly
    from math import gcd
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

def _resample(mono: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resample mono float32 audio from from_sr to to_sr."""
    if from_sr == to_sr:
        return mono
    if _HAS_SCIPY:
        try:
            g = gcd(from_sr, to_sr)
            return resample_poly(mono, to_sr // g, from_sr // g).astype(np.float32)
        except Exception as e:
            log.error(f"Resampling error with scipy: {e}")

    # Fallback numpy : interpolation linéaire (qualité suffisante pour la voix)
    target_len = int(len(mono) * to_sr / from_sr)
    indices = np.linspace(0, len(mono) - 1, target_len)
    return np.interp(indices, np.arange(len(mono)), mono).astype(np.float32)

def get_best_input_device():
    """Identifies the best available microphone on Windows with a scoring system."""
    try:
        devices = sd.query_devices()
        host_apis = sd.query_hostapis()

        best_idx = None
        best_score = -1
        selected_info = ""

        # Priority Keywords (Scoring based)
        high_priority = ["hands-free", "bluetooth", "jbl", "headset", "airpods", "hyperx", "steelseries", "logitech"]
        med_priority = ["usb microphone", "microphone array", "realtek"]
        ignore_keywords = ["hdmi", "nvidia", "stereo mix", "virtual", "output", "displayport"]

        for i, dev in enumerate(devices):
            if dev['max_input_channels'] <= 0:
                continue

            name = dev['name'].lower()
            if any(k in name for k in ignore_keywords):
                continue

            # API Scoring
            api_info = host_apis[dev['hostapi']]
            api_name = api_info['name'].upper()
            api_score = 0
            if "WASAPI" in api_name: api_score = 30
            elif "DIRECTSOUND" in api_name: api_score = 20
            elif "WDM-KS" in api_name: api_score = 10

            # Keyword Scoring
            kw_score = 0
            if any(k in name for k in high_priority): kw_score = 100
            elif any(k in name for k in med_priority): kw_score = 50
            else: kw_score = 10

            # Default Device Bonus (Reduced to ensure API preference)
            default_bonus = 15 if i == sd.default.device[0] else 0

            total_score = kw_score + api_score + default_bonus

            if total_score > best_score:
                best_score = total_score
                best_idx = i
                selected_info = f"{dev['name']} ({api_name}) | Score: {total_score}"

        if best_idx is not None:
            log.info(f"🎤 Micro sélectionné : {selected_info}")
            return best_idx

    except Exception as e:
        log.error(f"Erreur lors de la détection du micro : {e}")

    return sd.default.device[0]

async def audio_frame_generator(context: JarvisContext):
    """Captures microphone audio and yields PCM frames. Stabilized version."""
    loop = asyncio.get_running_loop()
    raw_q = queue.Queue(maxsize=200)
    proc_q = asyncio.Queue(maxsize=200)

    running = True

    def audio_worker():
        """Dedicated thread for audio processing (AGC, Resampling)."""
        nonlocal running
        smoothed_gain = 1.0

        while running:
            try:
                data_dict = raw_q.get(timeout=0.5)
                if data_dict is None: break

                mono = data_dict['data']
                native_sr = data_dict['sr']

                # 1. AGC (Slow/Smoothed)
                rms = np.sqrt(np.mean(mono ** 2))
                if rms > 0.001:
                    target_gain = AGC_TARGET_RMS / rms
                    target_gain = min(target_gain, AGC_MAX_GAIN)
                    # Smooth gain transition (approx 0.1s time constant)
                    smoothed_gain = 0.95 * smoothed_gain + 0.05 * target_gain
                else:
                    # Slowly return to unity gain during silence
                    smoothed_gain = 0.99 * smoothed_gain + 0.01 * 1.0

                mono = mono * smoothed_gain

                # 2. Resampling
                if native_sr != SAMPLE_RATE:
                    mono = _resample(mono, native_sr, SAMPLE_RATE)

                # 3. Safe PCM16 Conversion with Clipping
                # Ensure float32 is in range [-1, 1] before conversion
                mono_clipped = np.clip(mono, -1.0, 1.0)
                pcm16 = (mono_clipped * 32767).astype(np.int16).tobytes()

                # 4. Push to asyncio queue
                def _enqueue(d):
                    if proc_q.full():
                        try: proc_q.get_nowait()
                        except asyncio.QueueEmpty: pass
                    try: proc_q.put_nowait(d)
                    except Exception: pass

                loop.call_soon_threadsafe(_enqueue, pcm16)

            except queue.Empty:
                continue
            except Exception as e:
                log.error(f"Audio Worker Error: {e}")

    worker_thread = threading.Thread(target=audio_worker, daemon=True)
    worker_thread.start()

    retry_count = 0
    max_retries = 10
    last_device_id = None

    try:
      while retry_count < max_retries:
        try:
            device_id = get_best_input_device()
            dev_info = sd.query_devices(device_id)
            native_sr = int(dev_info['default_samplerate'])
            native_blocksize = int(FRAME_SIZE * native_sr / SAMPLE_RATE)

            log.info(f"🎙️ Audio Stream: {dev_info['name']} | Native SR: {native_sr}Hz")

            def callback(indata, frames, time_info, status):
                if status:
                    # Minimal logging in callback
                    pass

                # Lightweight capture
                mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
                try:
                    raw_q.put_nowait({'data': mono, 'sr': native_sr})
                except queue.Full:
                    pass

            stream = sd.InputStream(
                samplerate=native_sr,
                channels=1,
                dtype="float32",
                blocksize=native_blocksize,
                callback=callback,
                device=device_id
            )

            with stream:
                retry_count = 0
                last_device_id = device_id

                while True:
                    if sd.default.device[0] != last_device_id:
                        current_best = get_best_input_device()
                        if current_best != last_device_id:
                            log.info("🔄 Audio device change detected.")
                            break

                    try:
                        frame = await asyncio.wait_for(proc_q.get(), timeout=1.0)
                        if frame is None:
                            running = False
                            raw_q.put(None)
                            return
                        yield frame
                    except asyncio.TimeoutError:
                        if not stream.active: break
                        continue

        except Exception as e:
            retry_count += 1
            log.error(f"⚠️ Audio Error (retry {retry_count}): {e}")
            await asyncio.sleep(2)
    finally:
        running = False
        raw_q.put(None)
        log.info("🎤 Audio capture stopped.")

class VoiceActivityDetector:
    def __init__(self, model_path: str = "models/silero_vad.onnx"):
        if not os.path.exists(model_path):
            log.error(f"VAD Model missing: {model_path}")
            raise FileNotFoundError(model_path)

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self._reset_state()

    def _reset_state(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def is_speech(self, frame: bytes, threshold: float = VAD_THRESHOLD, context: JarvisContext = None) -> bool:
        """VAD with optimized threshold and barge-in protection."""
        if not frame: return False

        # Disable VAD while Jarvis is speaking (No barge-in for stability)
        if context and context.is_speaking:
            return False

        try:
            audio_int16 = np.frombuffer(frame, dtype=np.int16)
            # Safe conversion back to float for VAD
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            if len(audio_float32) != FRAME_SIZE:
                audio_float32 = np.pad(audio_float32, (0, FRAME_SIZE - len(audio_float32)))

            input_data = {
                "input": audio_float32[np.newaxis, :],
                "state": self._state,
                "sr": np.array([SAMPLE_RATE], dtype=np.int64)
            }

            out, stateN = self.session.run(None, input_data)
            self._state = stateN
            confidence = out.item()

            if confidence > threshold:
                log.info(f"🗣️ Speech detected (Conf: {confidence:.2f})")
                return True

            return False
        except Exception as e:
            log.error(f"VAD Error: {e}")
            self._reset_state()
            return False
