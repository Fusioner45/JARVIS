import asyncio
import numpy as np
import sounddevice as sd
import onnxruntime as ort
import os
import time
from jarvis.utils.config import SAMPLE_RATE, FRAME_SIZE
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
        g = gcd(from_sr, to_sr)
        return resample_poly(mono, to_sr // g, from_sr // g).astype(np.float32)
    else:
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

        high_priority = ["hands-free", "bluetooth", "jbl", "headset", "airpods", "hyperx", "steelseries", "logitech"]
        med_priority = ["usb microphone", "microphone array", "realtek"]
        ignore_keywords = ["hdmi", "nvidia", "stereo mix", "virtual", "output", "displayport"]

        for i, dev in enumerate(devices):
            if dev['max_input_channels'] <= 0:
                continue

            name = dev['name'].lower()
            if any(k in name for k in ignore_keywords):
                continue

            api_info = host_apis[dev['hostapi']]
            api_name = api_info['name'].upper()
            api_score = 0
            if "WASAPI" in api_name: api_score = 30
            elif "DIRECTSOUND" in api_name: api_score = 20
            elif "WDM-KS" in api_name: api_score = 10

            kw_score = 0
            if any(k in name for k in high_priority): kw_score = 100
            elif any(k in name for k in med_priority): kw_score = 50
            else: kw_score = 10

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
    """Captures microphone audio and yields PCM frames. Restored & Robust."""
    loop = asyncio.get_running_loop()
    q = asyncio.Queue(maxsize=1000)

    retry_count = 0
    max_retries = 10
    last_device_id = None
    frame = None

    while retry_count < max_retries:
        try:
            device_id = get_best_input_device()
            dev_info = sd.query_devices(device_id)
            native_sr = int(dev_info['default_samplerate'])

            # Correction : Forcer 16k si possible pour éviter resampling
            # Si le device supporte 16k, on l'utilise directement.
            try:
                sd.check_input_settings(device=device_id, samplerate=16000, channels=1)
                capture_sr = 16000
            except Exception:
                capture_sr = native_sr

            native_blocksize = int(FRAME_SIZE * capture_sr / SAMPLE_RATE)

            log.info(
                f"🎙️ Démarrage flux ("
                f"Device={device_id}, "
                f"CaptureSR={capture_sr}Hz, "
                f"TargetSR={SAMPLE_RATE}Hz, "
                f"Blocksize={native_blocksize}, "
                f"Latency={dev_info['default_low_input_latency']:.3f}s)"
            )

            def callback(indata, frames, time_info, status):
                if status:
                    log.warning(f"⚠️ Audio Status: {status}")

                if not loop.is_running():
                    return

                # Pipeline Mono Float32
                mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()

                # AGC minimal pour compenser les micros faibles
                amplitude = np.abs(mono).mean()
                if amplitude > 0.0001:
                    # AGC SOFT: Normalise vers 0.05 RMS
                    rms = np.sqrt(np.mean(mono ** 2))
                    if rms > 0.0001:
                        gain = 0.05 / rms
                        gain = min(gain, 10.0) # Gain max plus raisonnable
                        mono = np.clip(mono * gain, -1.0, 1.0)

                # Blocage is_speaking avec log
                if context.is_speaking:
                    if amplitude > 0.001:
                        log.info(f"🔇 Audio bloqué (is_speaking=True)")
                    return

                # Resampling
                if capture_sr != SAMPLE_RATE:
                    processed_mono = _resample(mono, capture_sr, SAMPLE_RATE)
                else:
                    processed_mono = mono

                def _enqueue(data):
                    if q.full():
                        try: q.get_nowait()
                        except asyncio.QueueEmpty: pass
                    try: q.put_nowait(data)
                    except Exception: pass

                pcm16 = (processed_mono * 32767).astype(np.int16).tobytes()
                loop.call_soon_threadsafe(_enqueue, pcm16)

            stream = sd.InputStream(
                samplerate=capture_sr,
                channels=1,
                dtype="float32",
                blocksize=native_blocksize,
                callback=callback,
                device=device_id
            )

            with stream:
                log.info("🎤 Microphone actif.")
                retry_count = 0
                last_device_id = device_id

                while True:
                    try:
                        if sd.default.device[0] != last_device_id:
                            current_best = get_best_input_device()
                            if current_best != last_device_id: break

                        frame = await asyncio.wait_for(q.get(), timeout=1.0)
                        if frame is None: return
                        yield frame
                    except asyncio.TimeoutError:
                        if not stream.active: break
                        continue

        except Exception as e:
            retry_count += 1
            log.error(f"⚠️ Erreur audio (retry {retry_count}/{max_retries}): {e}")
            await asyncio.sleep(3)

    log.critical("💀 Système audio KO.")

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

    def is_speech(self, frame: bytes, threshold: float = 0.05) -> bool:
        if not frame: return False

        try:
            audio_int16 = np.frombuffer(frame, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            rms = np.sqrt(np.mean(audio_float32**2))
            if rms < 0.00005: return False

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

            if confidence > 0.01:
                log.info(f"🔍 VAD: Conf={confidence:.3f} | RMS={rms:.6f}")

            if confidence > threshold:
                log.info(f"✅ VOICE DETECTED ({confidence:.2f})")
                return True

            return False
        except Exception as e:
            log.error(f"VAD Error: {e}")
            self._reset_state()
            return False
