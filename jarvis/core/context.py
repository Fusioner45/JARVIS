import asyncio
import queue
from dataclasses import dataclass, field
from jarvis.core.states import JarvisState
from jarvis.utils.logger import system_log as log

@dataclass
class JarvisContext:
    """Shared state for inter-module coordination with atomic control."""
    state: JarvisState = JarvisState.IDLE
    is_speaking: bool = False

    # Event loop reference for thread-safe calls
    loop: asyncio.AbstractEventLoop = field(default_factory=asyncio.get_event_loop)

    # Queues
    audio_output_queue: asyncio.Queue = field(default_factory=asyncio.Queue) # Phrases TTS (PCM)
    playback_sync_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=100)) # Chunks PCM (Sync)

    # Stop event for cancelling ongoing network streams (LLM/TTS)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)

    def set_state(self, new_state: JarvisState):
        self.state = new_state
        log.debug(f"State changed to: {new_state}")

    def reset_stop_event(self):
        self.stop_event.clear()

    def trigger_stop(self):
        """Atomsically signals to stop all current output activities."""
        self.stop_event.set()
        self.is_speaking = False

        # Clear queues
        while not self.playback_sync_queue.empty():
            try:
                self.playback_sync_queue.get_nowait()
            except queue.Empty:
                break

        # We can't easily clear an asyncio.Queue, but we can consume it
        while not self.audio_output_queue.empty():
            try:
                self.audio_output_queue.get_nowait()
                self.audio_output_queue.task_done()
            except asyncio.QueueEmpty:
                break

        log.info("🛑 Kill-Switch déclenché : Buffers vidés et signaux d'arrêt émis.")
