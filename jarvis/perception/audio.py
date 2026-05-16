import asyncio
import numpy as np
import sounddevice as sd
import onnxruntime as ort
import os
import time
from jarvis.utils.config import SAMPLE_RATE, FRAME_SIZE
from jarvis.utils.logger import audio_log as log
from jarvis.core.context import JarvisContext

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

            # Default Device Bonus
            default_bonus = 50 if i == sd.default.device[0] else 0

            total_score = kw_score + api_score + default_bonus

            if total_score > best_score:
                best_score = total_score
                best_idx = i
                selected_info = f"{dev['name']} ({api_name}) | Score: {total_score}"

        if best_idx is not None:
            log.info(f"🎤 Micro sélectionné : {selected_info}")
            return best_idx

    except Exception as e:
        log.error(f"Erreur lors de la sélection du micro : {e}")

    return sd.default.device[0]

async def audio_frame_generator(context: JarvisContext):
    """Captures microphone audio and yields PCM frames. Robust against disconnection."""
    loop = asyncio.get_running_loop()
    q = asyncio.Queue(maxsize=1000)

    retry_count = 0
    max_retries = 10
    last_device_id = None

    def callback(indata, frames, time_info, status):
        if status:
            log.warning(f"⚠️ Audio Status: {status}")

        if not loop.is_running():
            return

        mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()

        # Diagnostic: Live Amplitude
        amplitude = np.abs(mono).mean()
        if amplitude > 0.005:
            log.debug(f"📊 Mic Amp: {amplitude:.4f} {'(MUTED)' if context.is_speaking else ''}")

        if context.is_speaking:
            return

        def _enqueue(data):
            if q.full():
                try: q.get_nowait()
                except asyncio.QueueEmpty: pass
            try: q.put_nowait(data)
            except Exception: pass

        pcm16 = (mono * 32767).astype(np.int16).tobytes()
        loop.call_soon_threadsafe(_enqueue, pcm16)

    while retry_count < max_retries:
        try:
            device_id = get_best_input_device()
            dev_info = sd.query_devices(device_id)

            log.info(f"🎙️ Démarrage flux (Device={device_id}, API={dev_info['hostapi']}, SR={SAMPLE_RATE}Hz, Latency={dev_info['default_low_input_latency']:.3f}s)")

            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=FRAME_SIZE,
                callback=callback,
                device=device_id
            )

            with stream:
                log.info("🎤 Microphone actif - En attente...")
                retry_count = 0
                last_device_id = device_id

                while True:
                    try:
                        # Check if default device changed (hot-plug)
                        if sd.default.device[0] != last_device_id:
                            current_best = get_best_input_device()
                            if current_best != last_device_id:
                                log.info("🔄 Changement de périphérique détecté, redémarrage du flux...")
                                break

                        frame = await asyncio.wait_for(q.get(), timeout=1.0)
                        if frame is None: return # Global shutdown
                        yield frame
                    except asyncio.TimeoutError:
                        if not stream.active:
                            log.error("❌ Flux audio inactif.")
                            break
                        continue

        except Exception as e:
            retry_count += 1
            log.error(f"⚠️ Erreur audio (retry {retry_count}/{max_retries}): {e}")
            await asyncio.sleep(3)

    log.critical("💀 Système audio KO après multiples tentatives.")

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

    def is_speech(self, frame: bytes, threshold: float = 0.35) -> bool:
        """Robust VAD with lower threshold for hands-free mics."""
        if not frame: return False

        try:
            audio_int16 = np.frombuffer(frame, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_int16.astype(np.float32)**2)) / 32768.0

            # Lowered silence floor for low-gain headsets
            if rms < 0.0005: return False

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

            if confidence > 0.1:
                log.debug(f"🔍 VAD: Conf={confidence:.3f}, RMS={rms:.5f}")

            if confidence > threshold:
                log.info(f"🗣️ Parole détectée ({confidence:.2f})")
                return True

            return False
        except Exception as e:
            log.error(f"VAD Error: {e}")
            self._reset_state()
            return False
