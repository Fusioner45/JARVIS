import asyncio
from dataclasses import dataclass, field
from jarvis.core.states import JarvisState

@dataclass
class JarvisContext:
    """Minimal shared state injected into modules."""
    state: JarvisState = JarvisState.IDLE
    is_speaking: bool = False

    # Queue for audio chunks to be played back
    audio_output_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    def set_state(self, new_state: JarvisState):
        self.state = new_state
