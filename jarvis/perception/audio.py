import asyncio
import numpy as np
import sounddevice as sd
import onnxruntime as ort
import os
from jarvis.utils.config import SAMPLE_RATE, FRAME_SIZE
from jarvis.utils.logger import audio_log as log
from jarvis.core.context import JarvisContext

async def audio_frame_generator(context: JarvisContext):
    """Captures microphone audio and yields PCM frames. Discards if JARVIS is speaking."""
    loop = asyncio.get_running_loop()
    q = asyncio.Queue(maxsize=200)

    def callback(indata, frames, time, status):
        # Atomic security check: loop running? context valid?
        if not loop.is_running() or context.is_speaking:
            return

        try:
            if q.full():
                try: q.get_nowait()
                except asyncio.QueueEmpty: pass

            mono = indata[:, 0] if indata.ndim > 1 else indata
            pcm16 = (mono * 32767).astype(np.int16).tobytes()
            loop.call_soon_threadsafe(q.put_nowait, pcm16)
        except Exception: pass

    try:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                               blocksize=FRAME_SIZE, callback=callback)
        with stream:
            log.info("🎤 Microphone ouvert.")
            while True:
                try:
                    # Timeout serves as heartbeat and allows checking for exit
                    frame = await asyncio.wait_for(q.get(), timeout=1.0)

                    if frame is None: # Shutdown Sentinel
                        log.debug("Audio generator sentinel received.")
                        break

                    yield frame
                except asyncio.TimeoutError:
                    if not stream.active: break
                    continue
    except asyncio.CancelledError:
        log.info("Générateur audio stoppé.")
    except Exception as e:
        log.critical(f"Erreur flux audio : {e}")
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
        # Correct Silero VAD V4/V5 state shape
        self._state = np.zeros((2, 1, 64), dtype=np.float32)

    def is_speech(self, frame: bytes, threshold: float = 0.5) -> bool:
        if not frame: return False

        audio_int16 = np.frombuffer(frame, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_int16.astype(np.float32)**2)) / 32768.0
        if rms < 0.005: return False

        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        if len(audio_float32) != FRAME_SIZE:
            audio_float32 = np.pad(audio_float32, (0, FRAME_SIZE - len(audio_float32)))

        input_data = {
            "input": audio_float32[np.newaxis, :],
            "state": self._state,
            "sr": np.array([SAMPLE_RATE], dtype=np.int64)
        }

        try:
            out, stateN = self.session.run(None, input_data)
            self._state = stateN
            return out.item() > threshold
        except Exception as e:
            log.error(f"VAD Run Error: {e}")
            return False
