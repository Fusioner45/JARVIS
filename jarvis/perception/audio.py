import asyncio
import numpy as np
import sounddevice as sd
import onnxruntime as ort
import os
from jarvis.utils.config import SAMPLE_RATE, FRAME_SIZE
from jarvis.utils.logger import audio_log as log
from jarvis.core.context import JarvisContext

def get_best_input_device():
    """Identifies the best available microphone on Windows."""
    try:
        devices = sd.query_devices()
        default_in = sd.default.device[0]

        # 1. Check if default device is valid
        if default_in != -1:
            dev = devices[default_in]
            if dev['max_input_channels'] > 0:
                log.info(f"🎙️ Utilisation du périphérique par défaut : {dev['name']}")
                return default_in

        # 2. Priority Keywords
        priorities = ["microphone array", "realtek", "usb audio", "hyperx", "jabra", "steelseries"]
        for p in priorities:
            for i, dev in enumerate(devices):
                if p in dev['name'].lower() and dev['max_input_channels'] > 0:
                    log.info(f"🎙️ Périphérique prioritaire détecté : {dev['name']}")
                    return i

        # 3. Fallback to first available with input channels
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                log.info(f"🎙️ Fallback sur le premier périphérique disponible : {dev['name']}")
                return i

    except Exception as e:
        log.error(f"Erreur lors de la détection du micro : {e}")

    return None

async def audio_frame_generator(context: JarvisContext):
    """Captures microphone audio and yields PCM frames. Discards if JARVIS is speaking."""
    loop = asyncio.get_running_loop()
    q = asyncio.Queue(maxsize=500) # Increased buffer

    retry_count = 0
    max_retries = 3
    frame = None

    def callback(indata, frames, time, status):
        if status:
            log.warning(f"Audio status callback : {status}")

        if not loop.is_running():
            return

        # Still capture even if speaking for amplitude logs, but don't queue
        mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()

        # DIAGNOSTIC: Calculate real-time amplitude
        amplitude = np.abs(mono).mean()
        if amplitude > 0.01: # Log only if significant
            log.debug(f"📊 Amplitude Micro: {amplitude:.4f} {'(MUTÉ)' if context.is_speaking else ''}")

        if context.is_speaking:
            return

        def _enqueue(data):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(data)
            except Exception:
                pass

        pcm16 = (mono * 32767).astype(np.int16).tobytes()
        loop.call_soon_threadsafe(_enqueue, pcm16)

    try:
        while retry_count < max_retries:
            try:
                device_id = get_best_input_device()
                log.info(f"🎤 Initialisation flux audio (Device: {device_id}, Rate: {SAMPLE_RATE}Hz)...")

                stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    blocksize=FRAME_SIZE,
                    callback=callback,
                    device=device_id
                )

                with stream:
                    log.info("🎤 Microphone ouvert - En attente de parole...")
                    retry_count = 0 # Reset on success
                    while True:
                        try:
                            frame = await asyncio.wait_for(q.get(), timeout=1.0)
                            if frame is None:
                                break
                            yield frame
                        except asyncio.TimeoutError:
                            if not stream.active:
                                log.error("Flux audio inactif, tentative de redémarrage...")
                                break
                            continue

                if frame is None:
                    break # Shutdown requested

            except Exception as e:
                retry_count += 1
                log.error(f"Erreur flux audio (tentative {retry_count}/{max_retries}): {e}")
                await asyncio.sleep(2)

        if retry_count >= max_retries:
            log.critical("Le système audio a cessé de fonctionner après plusieurs tentatives.")

    except asyncio.CancelledError:
        log.info("Générateur audio stoppé.")
    except Exception as e:
        log.critical(f"Erreur critique générateur audio : {e}")
    finally:
        log.debug("Perception fermée.")

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
        # Correct Silero VAD state shape for latest versions (V5 expects 128)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def is_speech(self, frame: bytes, threshold: float = 0.4) -> bool:
        """Determines if a frame contains speech with robust logging."""
        if not frame:
            return False

        try:
            audio_int16 = np.frombuffer(frame, dtype=np.int16)

            # DIAGNOSTIC: RMS and Peak
            rms = np.sqrt(np.mean(audio_int16.astype(np.float32)**2)) / 32768.0

            # Even if below RMS threshold, we might want to see it in logs if it's close
            if rms < 0.001:
                return False

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

            if confidence > 0.1: # DEBUG: Show low confidence triggers in logs
                log.debug(f"🔍 VAD Diagnostic: Conf={confidence:.3f}, RMS={rms:.5f}")

            if confidence > threshold:
                log.info(f"🗣️ Parole détectée (Conf: {confidence:.2f})")
                return True

            return False
        except Exception as e:
            log.error(f"VAD Run Error: {e}")
            self._reset_state() # Auto-repair state on error
            return False
