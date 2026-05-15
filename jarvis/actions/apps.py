import os
import subprocess
import winreg
from jarvis.utils.config import APP_WHITELIST
from jarvis.utils.logger import action_log as log

class AppResolver:
    _cache = {}

    @classmethod
    def find_app(cls, app_name: str) -> str:
        """Finds application path via Whitelist, PATH, Registry, or Start Menu."""
        app_name = app_name.lower()
        if app_name in cls._cache: return cls._cache[app_name]

        # 1. Whitelist
        if app_name in APP_WHITELIST:
            res = APP_WHITELIST[app_name]
            cls._cache[app_name] = res
            return res

        # 2. System PATH
        try:
            path = subprocess.check_output(['where', app_name], stderr=subprocess.DEVNULL).decode().splitlines()[0]
            cls._cache[app_name] = path
            return path
        except: pass

        # 3. Windows Registry (App Paths)
        try:
            reg_path = f"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\{app_name}.exe"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path) as key:
                res, _ = winreg.QueryValueEx(key, "")
                cls._cache[app_name] = res
                return res
        except: pass

        # 4. Start Menu Scan
        menu_paths = [
            os.path.join(os.environ["ProgramData"], "Microsoft", "Windows", "Start Menu", "Programs"),
            os.path.join(os.environ["AppData"], "Microsoft", "Windows", "Start Menu", "Programs")
        ]
        for base in menu_paths:
            if not os.path.exists(base): continue
            for root, _, files in os.walk(base):
                for f in files:
                    if app_name in f.lower() and f.endswith(".lnk"):
                        res = os.path.join(root, f)
                        cls._cache[app_name] = res
                        return res

        return app_name # Fallback to name
