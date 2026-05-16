import asyncio
import time
from typing import List, Dict, Any
from jarvis.actions.registry import ActionRegistry
from jarvis.utils.logger import action_log as log, perf_tracker

class ActionGuard:
    def __init__(self, cooldown: float = 5.0, max_failures: int = 2):
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
                log.error(f"⛔ Action BANNED: {cmd_id}")
                return True
            else:
                del self.banned_until[cmd_id]
                self.failed_counts[cmd_id] = 0

        if cmd_id in self.last_execution:
            if now - self.last_execution[cmd_id] < self.cooldown:
                log.warning(f"⏳ Cooldown: {cmd_id}")
                return True

        if len(self.history) >= 3:
            recent_ids = [h['id'] for h in self.history[-3:]]
            if all(rid == cmd_id for rid in recent_ids):
                log.error(f"🛑 Loop Detected: {cmd_id}")
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
        else:
            self.failed_counts[cmd_id] = 0

class CommandExecutor:
    """Hardened Async Executor (Phase 8)."""

    def __init__(self, jarvis_ref=None):
        self.registry = ActionRegistry(jarvis_ref)
        self.guard = ActionGuard()

    async def execute(self, name: str, args: List[str]) -> str:
        """Executes action off-thread if it's a blocking IO/System call."""
        cmd_id = f"{name}({args})"
        action = self.registry.get_action(name)

        if not action:
            return f"FAILED: {name} unknown."

        # Security Sanitization
        forbidden = [';', '&', '|', '$', '`']
        for arg in args:
            if any(c in str(arg) for c in forbidden):
                log.critical(f"🚨 INJECTION BLOCKED: {cmd_id}")
                return "FAILED: Security violation."

        if self.guard.is_blocked(cmd_id):
            return f"FAILED: Cooldown or loop."

        try:
            loop = asyncio.get_running_loop()

            # Offload blocking registry calls to ThreadPool
            with perf_tracker(log, f"ASYNC_EXEC: {name}"):
                # Note: Some actions might return "HANDLED_ASYNC_..." which is fine.
                res = await loop.run_in_executor(None, action, *args)

                success = "ERREUR" not in res.upper() and "FAILED" not in res.upper()
                self.guard.record_result(cmd_id, success)

                if res in ["HANDLED_ASYNC_HA", "HANDLED_ASYNC_VISION"]:
                    return res

                return f"SUCCESS: {res}" if success else f"FAILED: {res}"
        except Exception as e:
            log.error(f"Executor Error {name}: {e}")
            self.guard.record_result(cmd_id, False)
            return f"FAILED: {str(e)}"
