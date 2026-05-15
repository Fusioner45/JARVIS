import os
import webbrowser
import threading
import time
import pyautogui
import pygetwindow as gw
from typing import List, Callable, Dict, Any
from jarvis.actions.apps import AppResolver
from jarvis.utils.logger import action_log as log
from jarvis.utils.config import PLAYLISTS

class ActionRegistry:
    """Type-safe registry for ALL validated JARVIS actions (Phase 2)."""

    def __init__(self, jarvis_ref=None):
        self.jarvis = jarvis_ref
        self._actions: Dict[str, Callable] = {}
        self._register_defaults()

    def register(self, name: str, func: Callable):
        self._actions[name.upper()] = func

    def get_action(self, name: str) -> Callable:
        return self._actions.get(name.upper())

    def _register_defaults(self):
        # Core Apps
        self.register("OPEN_APP", self._open_app)
        self.register("SEARCH_WEB", self._search_web)
        self.register("PLAY_MUSIC", self._play_music)

        # System Info
        self.register("GET_GPU_TEMP", self._get_gpu_temp)
        self.register("SPLIT_SCREEN", self._split_screen)

        # Memory Tools
        self.register("SAVE_FACT", self._save_fact)
        self.register("DELETE_FACT", self._delete_fact)
        self.register("INDEX_PDF", self._index_pdf)

        # IoT & Vision
        self.register("HA_CONTROL", self._ha_control)
        self.register("SCREENSHOT_ANALYZE", self._screenshot_analyze)

    def _open_app(self, app_name: str) -> str:
        target = AppResolver.find_app(app_name)
        if os.path.exists(target) or target.endswith(".exe") or target.endswith(".lnk"):
            os.startfile(target)
            return f"Application {app_name} lancée."
        return f"ERREUR: Application {app_name} introuvable."

    def _search_web(self, query: str) -> str:
        webbrowser.open(f"https://www.google.com/search?q={query}")
        return f"Recherche web lancée pour : {query}"

    def _play_music(self, query: str) -> str:
        query = query.lower()
        url = next((u for k, u in PLAYLISTS.items() if k in query),
                   f"https://www.youtube.com/results?search_query={query}")
        webbrowser.open(url)
        threading.Thread(target=lambda: (time.sleep(5), pyautogui.press('space')), daemon=True).start()
        return f"Lecture lancée sur {url}"

    def _get_gpu_temp(self) -> str:
        if self.jarvis and hasattr(self.jarvis, 'gpu_mon'):
            temp = self.jarvis.gpu_mon.get_temperature()
            return f"Température GPU : {temp}°C"
        return "ERREUR: Moniteur GPU indisponible."

    def _split_screen(self, app1: str, app2: str) -> str:
        try:
            windows = gw.getAllWindows()
            w1 = [w for w in windows if app1.lower() in w.title.lower()]
            w2 = [w for w in windows if app2.lower() in w.title.lower()]
            if w1 and w2:
                w1[0].restore(); w1[0].moveTo(0, 0); w1[0].resizeTo(960, 1080)
                w2[0].restore(); w2[0].moveTo(960, 0); w2[0].resizeTo(960, 1080)
                return "Écran scindé avec succès."
            return "ERREUR: Fenêtres non trouvées."
        except Exception as e: return f"ERREUR: {e}"

    def _save_fact(self, fact_type: str, content: str) -> str:
        if self.jarvis and hasattr(self.jarvis, 'memory'):
            self.jarvis.memory.save_memory(fact_type, content)
            return f"Fait '{fact_type}' mémorisé."
        return "ERREUR: Mémoire indisponible."

    def _delete_fact(self, term: str) -> str:
        if self.jarvis and hasattr(self.jarvis, 'memory'):
            self.jarvis.memory.delete_memory(term)
            return f"Fait lié à '{term}' supprimé."
        return "ERREUR: Mémoire indisponible."

    def _index_pdf(self, path: str) -> str:
        if self.jarvis and hasattr(self.jarvis, 'memory'):
            self.jarvis.memory.index_pdf(path)
            return f"PDF {path} indexé."
        return "ERREUR: Indexeur indisponible."

    def _ha_control(self, entity: str, service: str) -> str:
        # Appelé en async par l'orchestrateur, ici on prépare juste le signal
        return "HANDLED_ASYNC_HA"

    def _screenshot_analyze(self) -> str:
        # Appelé en async par l'orchestrateur
        return "HANDLED_ASYNC_VISION"
