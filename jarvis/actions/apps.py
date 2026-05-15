import os
import subprocess
import winreg
from jarvis.utils.config import APP_WHITELIST
from jarvis.utils.logger import system_log as log

class AppResolver:
    _cache = {}

    @classmethod
    def find_app(cls, app_name: str) -> str:
        """Résout le chemin d'une application via Whitelist, PATH, Registre et Menu Démarrer."""
        app_name = app_name.lower()
        if app_name in cls._cache:
            return cls._cache[app_name]

        # 1. Whitelist
        if app_name in APP_WHITELIST:
            res = APP_WHITELIST[app_name]
            cls._cache[app_name] = res
            return res

        # 2. PATH (where)
        try:
            path = subprocess.check_output(['where', app_name], stderr=subprocess.DEVNULL).decode().splitlines()[0]
            cls._cache[app_name] = path
            return path
        except: pass

        # 3. Registre Windows (App Paths)
        try:
            reg_path = f"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\{app_name}.exe"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path) as key:
                res, _ = winreg.QueryValueEx(key, "")
                cls._cache[app_name] = res
                return res
        except: pass

        # 4. Start Menu Scan
        start_menu_paths = [
            os.path.join(os.environ["ProgramData"], "Microsoft", "Windows", "Start Menu", "Programs"),
            os.path.join(os.environ["AppData"], "Microsoft", "Windows", "Start Menu", "Programs")
        ]

        for base in start_menu_paths:
            if not os.path.exists(base): continue
            for root, dirs, files in os.walk(base):
                for f in files:
                    if app_name in f.lower() and f.endswith(".lnk"):
                        cls._cache[app_name] = os.path.join(root, f)
                        return os.path.join(root, f)

        log.warning(f"Application non résolue : {app_name}")
        return app_name # Fallback
