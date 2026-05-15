import asyncio
from dataclasses import dataclass, field
from jarvis.core.states import JarvisState

@dataclass
class JarvisContext:
    """Shared state for inter-module coordination."""
    state: JarvisState = JarvisState.IDLE
    is_speaking: bool = False

    # Asyncio queue for TTS audio chunks
    audio_output_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    def set_state(self, new_state: JarvisState):
        self.state = new_state
