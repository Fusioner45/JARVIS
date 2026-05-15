import time
import logging
from typing import List, Dict, Any, Callable
from jarvis.actions.apps import AppResolver
from jarvis.actions.registry import ActionRegistry
from jarvis.utils.logger import action_log as log, perf_tracker

class ActionGuard:
    """Prevents execution loops and bans failing commands."""
    def __init__(self, cooldown: float = 10.0, max_failures: int = 2):
        self.cooldown = cooldown
        self.max_failures = max_failures
        self.history: List[Dict[str, Any]] = []
        self.last_execution: Dict[str, float] = {}
        self.failed_counts: Dict[str, int] = {}
        self.banned_until: Dict[str, float] = {}

    def is_blocked(self, cmd_id: str) -> bool:
        now = time.time()
        if cmd_id in self.banned_until:
            if now < self.banned_until[cmd_id]:
                log.error(f"⛔ Commande bannie (Échecs répétés) : {cmd_id}")
                return True
            else:
                del self.banned_until[cmd_id]
                self.failed_counts[cmd_id] = 0

        if cmd_id in self.last_execution:
            if now - self.last_execution[cmd_id] < self.cooldown:
                log.warning(f"⏳ Cooldown actif (10s) pour : {cmd_id}")
                return True

        if len(self.history) >= 3:
            recent_ids = [h['id'] for h in self.history[-3:]]
            if all(rid == cmd_id for rid in recent_ids):
                log.error(f"🛑 Boucle infinie détectée pour : {cmd_id}")
                return True
        return False

    def record_result(self, cmd_id: str, success: bool):
        now = time.time()
        self.last_execution[cmd_id] = now
        self.history.append({'id': cmd_id, 'ts': now, 'success': success})
        if not success:
            self.failed_counts[cmd_id] = self.failed_counts.get(cmd_id, 0) + 1
            if self.failed_counts[cmd_id] >= self.max_failures:
                self.banned_until[cmd_id] = now + 300
                log.error(f"🚫 Commande {cmd_id} bannie pour 5 minutes.")
        else:
            self.failed_counts[cmd_id] = 0

class CommandExecutor:
    """Zero-Trust executor using the type-safe ActionRegistry and ActionGuard."""

    def __init__(self):
        self.registry = ActionRegistry()
        self.guard = ActionGuard()

    def execute(self, name: str, args: List[str]) -> str:
        cmd_id = f"{name}({args})"

        if self.guard.is_blocked(cmd_id):
            return f"FAILED: Action {name} is currently blocked (cooldown or ban)."

        action = self.registry.get_action(name)
        if not action:
            log.warning(f"⚠️ Action inconnue : {name}")
            return f"FAILED: {name} not in registry."

        # Basic security sanitization
        if any(any(c in str(a) for c in [';', '&', '|', '$', '`']) for a in args):
            return "FAILED: Security violation (injection detected)."

        try:
            with perf_tracker(log, f"Action {name}"):
                result = action(*args)
                success = "ERREUR" not in result.upper() and "FAILED" not in result.upper()
                self.guard.record_result(cmd_id, success)
                return f"SUCCESS: {result}" if success else f"FAILED: {result}"
        except Exception as e:
            log.error(f"Erreur exécution {name} : {e}")
            self.guard.record_result(cmd_id, False)
            return f"FAILED: {str(e)}"
