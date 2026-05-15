from typing import List
from jarvis.actions.registry import ActionRegistry
from jarvis.utils.logger import action_log as log, perf_tracker

class CommandExecutor:
    """Zero-Trust executor using the type-safe ActionRegistry."""

    def __init__(self):
        self.registry = ActionRegistry()

    def execute(self, name: str, args: List[str]) -> str:
        action = self.registry.get_action(name)
        if not action:
            log.warning(f"⚠️ Action non autorisée ou inconnue : {name}")
            return f"FAILED: {name} not in registry"

        # Basic security sanitization
        if any(any(c in str(a) for c in [';', '&', '|', '$', '`']) for a in args):
            return "FAILED: Security violation (injection detected)"

        try:
            with perf_tracker(log, f"Action {name}"):
                return action(*args)
        except Exception as e:
            log.error(f"Erreur exécution {name} : {e}")
            return f"FAILED: {e}"
