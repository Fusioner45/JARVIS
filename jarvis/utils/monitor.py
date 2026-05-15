import asyncio
import time
import psutil
import os
from jarvis.utils.logger import system_log as log

class RuntimeSupervisor:
    """Production Runtime Supervision (V7.2)."""

    def __init__(self, context, interval: float = 10.0):
        self.context = context
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self._running = False
        self._lag_threshold = 1.0 # 1s
        self._tasks_watchdog = {}

    async def start(self):
        self._running = True
        log.info("🛡️ Runtime Supervisor V7.2 active.")
        while self._running:
            start_check = time.perf_counter()
            await asyncio.sleep(self.interval)

            # 1. Loop Lag
            actual_sleep = time.perf_counter() - start_check
            lag = actual_sleep - self.interval
            if lag > self._lag_threshold:
                log.warning(f"⚠️ Event Loop Lag: {lag*1000:.1f}ms")

            # 2. Queue Health
            out_q = self.context.audio_output_queue.qsize()
            sync_q = self.context.playback_sync_queue.qsize()
            if out_q > 3 or sync_q > 90:
                log.warning(f"🚨 Backpressure detected: TTS Queue={out_q}, Playback={sync_q}")

            # 3. Task Health
            self.check_stuck_tasks()

            # 4. Memory Monitoring
            mem_rss = self.process.memory_info().rss / (1024 * 1024)
            if mem_rss > 1500: # 1.5GB warning
                log.warning(f"📈 High Memory Usage: {mem_rss:.1f}MB")

            log.debug(f"SysHealth | RAM: {mem_rss:.1f}MB | Active Tasks: {len(asyncio.all_tasks())}")

    def stop(self):
        self._running = False

    def track_task(self, name: str, task: asyncio.Task):
        self._tasks_watchdog[name] = {
            "task": task,
            "start": time.time()
        }
        task.add_done_callback(lambda t: self._on_task_done(name, t))

    def _on_task_done(self, name, task):
        if name in self._tasks_watchdog:
            try:
                if task.cancelled():
                    log.debug(f"Task '{name}' was cancelled.")
                elif task.exception():
                    log.error(f"Task '{name}' failed with exception: {task.exception()}")

                duration = time.time() - self._tasks_watchdog[name]["start"]
                if duration > 10.0:
                    log.info(f"Slow Task Complete: {name} took {duration:.1f}s")
            except: pass
            del self._tasks_watchdog[name]

    def check_stuck_tasks(self):
        now = time.time()
        for name, info in list(self._tasks_watchdog.items()):
            if not info["task"].done():
                duration = now - info["start"]
                # Certain tasks like 'supervisor' or 'main_loop' are expected to run long
                if duration > 60.0 and name not in ["supervisor", "playback_manager", "main_loop"]:
                    log.warning(f"💀 Potentially Stuck Task: {name} ({duration:.1f}s)")
