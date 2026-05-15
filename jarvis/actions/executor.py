import os
import subprocess
import time
import webbrowser
import threading
import pyautogui
import pygetwindow as gw
from typing import List, Dict, Any
from jarvis.actions.apps import AppResolver
from jarvis.utils.logger import action_log as log
from jarvis.utils.config import PLAYLISTS

class ActionRegistry:
    ALLOWED_COMMANDS = {
        "OPEN_APP", "SEARCH_WEB", "PLAY_MUSIC", "SAVE_FACT",
        "DELETE_FACT", "SAVE_TASK", "GET_GPU_TEMP", "SPLIT_SCREEN",
        "WORK_MODE", "GET_WEATHER", "CALC_TRIP", "SET_VOICE_MORPH",
        "INDEX_PDF", "SCREENSHOT_ANALYZE", "HA_CONTROL"
    }

    @classmethod
    def is_allowed(cls, cmd_name: str) -> bool:
        return cmd_name.upper() in cls.ALLOWED_COMMANDS

class ActionGuard:
    def __init__(self, cooldown: float = 10.0):
        self.cooldown = cooldown
        self.history = []
        self.last_execution = {}
        self.failed_counts = {}

    def is_blocked(self, cmd_id: str) -> bool:
        now = time.time()
        if self.failed_counts.get(cmd_id, 0) >= 2:
            return True
        if cmd_id in self.last_execution:
            if now - self.last_execution[cmd_id] < self.cooldown:
                return True
        return False

    def record(self, cmd_id: str, success: bool):
        self.last_execution[cmd_id] = time.time()
        self.history.append((cmd_id, success))
        if not success:
            self.failed_counts[cmd_id] = self.failed_counts.get(cmd_id, 0) + 1
        else:
            self.failed_counts[cmd_id] = 0

class CommandExecutor:
    @staticmethod
    def execute(cmd_name: str, args: List[str], jarvis_ref=None) -> str:
        if not ActionRegistry.is_allowed(cmd_name):
            return f"FAILED: {cmd_name} not allowed"

        forbidden = [';', '&', '|', '$', '>', '<', '`']
        for arg in args:
            if any(c in str(arg) for c in forbidden):
                return "FAILED: Security violation"

        try:
            if cmd_name == "OPEN_APP":
                app_id = args[0].lower()
                if app_id == "youtube":
                    webbrowser.open("https://youtube.com")
                    return "SUCCESS: YouTube opened"
                target = AppResolver.find_app(app_id)
                if os.path.exists(target):
                    os.startfile(target)
                    return f"SUCCESS: {app_id} lancé"
                return f"FAILED: {app_id} non trouvé"

            elif cmd_name == "SEARCH_WEB":
                webbrowser.open(f"https://www.google.com/search?q={args[0]}")
                return "SUCCESS"

            elif cmd_name == "PLAY_MUSIC":
                query = args[0].lower()
                target_url = None
                for kw, url in PLAYLISTS.items():
                    if kw in query: target_url = url; break

                if not target_url and any(w in query for w in ["local", "mon pc"]):
                    os.startfile(r'C:\musique')
                    return "SUCCESS: Local music folder opened"

                url = target_url or f"https://www.youtube.com/results?search_query={args[0]}"
                webbrowser.open(url)
                # Auto-play simulation
                threading.Thread(target=lambda: (time.sleep(5), pyautogui.press('space')), daemon=True).start()
                return f"SUCCESS: Playing music on {url}"

            elif cmd_name == "GET_GPU_TEMP":
                if jarvis_ref:
                    temp = jarvis_ref.gpu_mon.get_temperature()
                    return f"SUCCESS: GPU is at {temp}°C"
                return "FAILED: GPU monitor unavailable"

            elif cmd_name == "SPLIT_SCREEN":
                try:
                    windows = gw.getAllWindows()
                    w1 = [w for w in windows if args[0].lower() in w.title.lower()]
                    w2 = [w for w in windows if args[1].lower() in w.title.lower()]
                    if w1 and w2:
                        w1[0].restore(); w1[0].moveTo(0, 0); w1[0].resizeTo(960, 1080)
                        w2[0].restore(); w2[0].moveTo(960, 0); w2[0].resizeTo(960, 1080)
                        return "SUCCESS: Windows split"
                    return "FAILED: Windows not found"
                except Exception as e: return f"FAILED: {e}"

            elif cmd_name == "GET_WEATHER":
                webbrowser.open(f"https://www.google.com/search?q=meteo+{args[0]}")
                return "SUCCESS"

            elif cmd_name == "CALC_TRIP":
                webbrowser.open(f"https://www.google.com/maps/dir/{args[0]}/{args[1]}")
                return "SUCCESS"

            return "HANDLED_BY_ORCHESTRATOR"

        except Exception as e:
            log.error(f"Error {cmd_name}: {e}")
            return f"FAILED: {e}"
