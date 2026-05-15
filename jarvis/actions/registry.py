import os
import webbrowser
import threading
import time
import pyautogui
from typing import List, Callable, Dict
from jarvis.actions.apps import AppResolver
from jarvis.utils.logger import action_log as log
from jarvis.utils.config import PLAYLISTS

class ActionRegistry:
    """Type-safe registry for validated JARVIS actions."""

    def __init__(self):
        self._actions: Dict[str, Callable] = {}
        self._register_defaults()

    def register(self, name: str, func: Callable):
        self._actions[name.upper()] = func

    def get_action(self, name: str) -> Callable:
        return self._actions.get(name.upper())

    def _register_defaults(self):
        self.register("OPEN_APP", self._open_app)
        self.register("SEARCH_WEB", self._search_web)
        self.register("PLAY_MUSIC", self._play_music)

    def _open_app(self, app_name: str) -> str:
        target = AppResolver.find_app(app_name)
        if os.path.exists(target) or target.endswith(".exe") or target.endswith(".lnk"):
            os.startfile(target)
            return f"Application {app_name} lancée avec succès."
        return f"Erreur : Impossible de trouver l'application {app_name}."

    def _search_web(self, query: str) -> str:
        webbrowser.open(f"https://www.google.com/search?q={query}")
        return f"Recherche Google lancée pour : {query}"

    def _play_music(self, query: str) -> str:
        query = query.lower()
        url = next((u for k, u in PLAYLISTS.items() if k in query),
                   f"https://www.youtube.com/results?search_query={query}")
        webbrowser.open(url)
        threading.Thread(target=lambda: (time.sleep(5), pyautogui.press('space')), daemon=True).start()
        return f"Musique lancée sur {url}"
