import os
import subprocess
import time
import webbrowser
import threading
import pyautogui
from typing import List
from jarvis.actions.apps import AppResolver
from jarvis.utils.logger import action_log as log, perf_tracker
from jarvis.utils.config import PLAYLISTS

class ActionRegistry:
    ALLOWED = {
        "OPEN_APP", "SEARCH_WEB", "PLAY_MUSIC", "SAVE_FACT", "DELETE_FACT",
        "SAVE_TASK", "GET_GPU_TEMP", "SPLIT_SCREEN", "WORK_MODE",
        "INDEX_PDF", "SCREENSHOT_ANALYZE", "HA_CONTROL"
    }

    @classmethod
    def is_allowed(cls, name: str) -> bool:
        return name.upper() in cls.ALLOWED

class CommandExecutor:
    @staticmethod
    def execute(name: str, args: List[str], jarvis_ref=None) -> str:
        if not ActionRegistry.is_allowed(name): return f"FAILED: {name} forbidden"

        # Security check
        if any(any(c in str(a) for c in [';', '&', '|', '$']) for a in args):
            return "FAILED: Security violation"

        log.info(f"⚡ Exécution : {name}({args})")
        try:
            with perf_tracker(log, f"Action {name}"):
                if name == "OPEN_APP":
                    target = AppResolver.find_app(args[0])
                    os.startfile(target)
                    return "SUCCESS"

                elif name == "SEARCH_WEB":
                    webbrowser.open(f"https://www.google.com/search?q={args[0]}")
                    return "SUCCESS"

                elif name == "PLAY_MUSIC":
                    query = args[0].lower()
                    url = next((u for k, u in PLAYLISTS.items() if k in query),
                               f"https://www.youtube.com/results?search_query={args[0]}")
                    webbrowser.open(url)
                    threading.Thread(target=lambda: (time.sleep(5), pyautogui.press('space')), daemon=True).start()
                    return "SUCCESS"

                return "HANDLED_BY_ORCHESTRATOR"
        except Exception as e:
            log.error(f"Action Error {name}: {e}")
            return f"FAILED: {e}"
