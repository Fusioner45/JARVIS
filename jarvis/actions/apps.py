import os
import shutil
from typing import Dict, Optional
from jarvis.utils.logger import action_log as log

class AppResolver:
    """Resolves Windows applications using Registry, PATH, and Whitelist (Phase 5)."""

    _cache: Dict[str, str] = {}

    @classmethod
    def find_app(cls, name: str) -> str:
        name = name.lower()
        if name in cls._cache:
            return cls._cache[name]

        # 1. Whitelist
        from jarvis.utils.config import APP_WHITELIST
        if name in APP_WHITELIST:
            path = APP_WHITELIST[name]
            cls._cache[name] = path
            return path

        # 2. System PATH
        path = shutil.which(name)
        if path:
            cls._cache[name] = path
            return path

        # 3. Windows Registry (App Paths)
        reg_path = cls._search_registry(name)
        if reg_path:
            cls._cache[name] = reg_path
            return reg_path

        # 4. Start Menu / common locations (Fallback to just name if not found)
        log.warning(f"Application '{name}' non trouvée via Registry/PATH.")
        return name

    @staticmethod
    def _search_registry(name: str) -> Optional[str]:
        """Looks up executable path in Windows Registry."""
        if os.name != 'nt':
            return None

        try:
            import winreg
        except ImportError:
            return None

        # Simple suffix addition for common apps
        search_names = [name, f"{name}.exe"]
        keys = [
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths"
        ]

        for s_name in search_names:
            for root_key in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
                for key_base in keys:
                    try:
                        full_key = f"{key_base}\\{s_name}"
                        with winreg.OpenKey(root_key, full_key) as k:
                            path, _ = winreg.QueryValueEx(k, "")
                            if os.path.exists(path):
                                return path
                    except (FileNotFoundError, OSError):
                        continue
        return None
