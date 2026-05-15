import time
import logging
from typing import List, Dict, Any, Callable
from jarvis.actions.registry import ActionRegistry
from jarvis.utils.logger import action_log as log, perf_tracker

class ActionGuard:
    """Prevents execution loops and bans failing commands (Phase 2)."""
    def __init__(self, cooldown: float = 10.0, max_failures: int = 2):
        self.cooldown = cooldown
        self.max_failures = max_failures
        self.history: List[Dict[str, Any]] = []
        self.last_execution: Dict[str, float] = {}
        self.failed_counts: Dict[str, int] = {}
        self.banned_until: Dict[str, float] = {}

    def is_blocked(self, cmd_id: str) -> bool:
        now = time.time()

        # Ban Check
        if cmd_id in self.banned_until:
            if now < self.banned_until[cmd_id]:
                log.error(f"⛔ Commande BANNIE (Échecs) : {cmd_id}")
                return True
            else:
                del self.banned_until[cmd_id]
                self.failed_counts[cmd_id] = 0

        # Cooldown Check
        if cmd_id in self.last_execution:
            if now - self.last_execution[cmd_id] < self.cooldown:
                log.warning(f"⏳ Cooldown (Action Registry Protection) : {cmd_id}")
                return True

        # Infinity loop check (Last 3 actions)
        if len(self.history) >= 3:
            recent_ids = [h['id'] for h in self.history[-3:]]
            if all(rid == cmd_id for rid in recent_ids):
                log.error(f"🛑 Boucle infinie STOPPÉE pour : {cmd_id}")
                return True

        return False

    def record_result(self, cmd_id: str, success: bool):
        now = time.time()
        self.last_execution[cmd_id] = now
        self.history.append({'id': cmd_id, 'ts': now, 'success': success})

        if not success:
            self.failed_counts[cmd_id] = self.failed_counts.get(cmd_id, 0) + 1
            if self.failed_counts[cmd_id] >= self.max_failures:
                self.banned_until[cmd_id] = now + 300 # 5 min
        else:
            self.failed_counts[cmd_id] = 0

class CommandExecutor:
    """Zero-Trust executor using the type-safe ActionRegistry (Phase 2)."""

    def __init__(self, jarvis_ref=None):
        self.registry = ActionRegistry(jarvis_ref)
        self.guard = ActionGuard()

    def execute(self, name: str, args: List[str]) -> str:
        cmd_id = f"{name}({args})"

        # 1. ActionRegistry Validation
        action = self.registry.get_action(name)
        if not action:
            log.error(f"⚠️ Tentative d'exécution non enregistrée : {name}")
            return f"FAILED: {name} not in registry (Zero-Trust violation)"

        # 2. Argument Sanitization
        forbidden = [';', '&', '|', '$', '`', '<', '>']
        for arg in args:
            if any(c in str(arg) for c in forbidden):
                log.critical(f"🚨 TENTATIVE D'INJECTION DÉTECTÉE : {cmd_id}")
                return "FAILED: Security violation (shell injection detected)"

        # 3. Guardrail (Anti-Loop)
        if self.guard.is_blocked(cmd_id):
            return f"FAILED: Action {name} is blocked (Cooldown or Loop detected)."

        try:
            with perf_tracker(log, f"EXEC: {name}"):
                # 4. Schema check (Simple count for now)
                # On pourrait ajouter une validation de types plus tard
                res = action(*args)

                success = "ERREUR" not in res.upper() and "FAILED" not in res.upper()
                self.guard.record_result(cmd_id, success)

                if res in ["HANDLED_ASYNC_HA", "HANDLED_ASYNC_VISION"]:
                    return res

                return f"SUCCESS: {res}" if success else f"FAILED: {res}"
        except Exception as e:
            log.error(f"Execution Exception {name}: {e}")
            self.guard.record_result(cmd_id, False)
            return f"FAILED: Runtime error {str(e)}"
