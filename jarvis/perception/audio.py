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
        if status:
            # Non-blocking log if possible
            pass

        # Security: check if loop is running
        if not loop.is_running():
            return

        if context.is_speaking:
            return

        try:
            # Atomic overflow handling
            if q.full():
                try: q.get_nowait()
                except asyncio.QueueEmpty: pass

            mono = indata[:, 0] if indata.ndim > 1 else indata
            pcm16 = (mono * 32767).astype(np.int16).tobytes()
            loop.call_soon_threadsafe(q.put_nowait, pcm16)
        except Exception:
            pass

    try:
        # Use explicit device if needed, here we use default
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                               blocksize=FRAME_SIZE, callback=callback)
        with stream:
            log.info("🎤 Flux audio initialisé.")
            while True:
                try:
                    # Timeout as heart-beat and cancellation check
                    frame = await asyncio.wait_for(q.get(), timeout=1.0)
                    if frame is None: # Explicit sentinel for shutdown
                        log.debug("Sentinel received in audio generator.")
                        break
                    yield frame
                except asyncio.TimeoutError:
                    # Check if we should exit even without sentinel
                    if not stream.active:
                        break
                    continue
    except asyncio.CancelledError:
        log.info("Générateur audio annulé.")
    except Exception as e:
        log.critical(f"Critical Audio Pipe Error: {e}")
    finally:
        log.debug("Audio frame generator closed.")

class VoiceActivityDetector:
    def __init__(self, model_path: str = "models/silero_vad.onnx"):
        if not os.path.exists(model_path):
            log.error(f"VAD Model not found at {model_path}")
            raise FileNotFoundError(model_path)

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self._reset_state()

    def _reset_state(self):
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

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
            "sr": np.array([SAMPLE_RATE], dtype=np.int64),
            "h": self._h,
            "c": self._c
        }

        try:
            out, h, c = self.session.run(None, input_data)
            self._h, self._c = h, c
            return out.item() > threshold
        except Exception as e:
            log.error(f"VAD Error: {e}")
            return False
