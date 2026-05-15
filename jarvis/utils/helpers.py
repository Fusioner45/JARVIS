import logging
import subprocess
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

log = logging.getLogger("Jarvis.Utils")

class AudioController:
    def __init__(self):
        from ctypes import cast, POINTER
        try:
            devices = AudioUtilities.GetSpeakers()
            self.interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume = cast(self.interface, POINTER(IAudioEndpointVolume))
        except Exception as e:
            log.warning(f"AudioController failed: {e}. Using PowerShell Fallback.")
            self.volume = None

    def set_ducking(self, duck: bool):
        if self.volume:
            try:
                target = 0.2 if duck else 1.0 # 20% volume during listening
                self.volume.SetMasterVolumeLevelScalar(target, None)
                return
            except Exception:
                pass

        try:
            vol = 20 if duck else 100
            subprocess.run(["powershell", "-Command", f"(Get-WmiObject -Class Win32_AudioControl).SetVolume({vol})"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

class GpuMonitor:
    def __init__(self):
        self.enabled = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.enabled = True
        except (ImportError, Exception):
            log.debug("NVIDIA Management Library (pynvml) not available.")

    def get_temperature(self) -> int:
        if self.enabled:
            try:
                import pynvml
                return pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                return -1
        return -1
