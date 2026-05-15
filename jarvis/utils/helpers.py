import logging
import subprocess
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import pynvml

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
                target = 0.1 if duck else 1.0
                self.volume.SetMasterVolumeLevelScalar(target, None)
                return
            except Exception:
                pass

        try:
            vol = 10 if duck else 100
            subprocess.run(["powershell", "-Command", f"(Get-WmiObject -Class Win32_AudioControl).SetVolume({vol})"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

class GpuMonitor:
    def __init__(self):
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.enabled = True
        except Exception:
            self.enabled = False

    def get_temperature(self) -> int:
        if self.enabled:
            return pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
        return -1
