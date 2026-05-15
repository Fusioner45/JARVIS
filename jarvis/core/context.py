import asyncio
from dataclasses import dataclass, field
from jarvis.core.states import JarvisState

@dataclass
class JarvisContext:
    state: JarvisState = JarvisState.IDLE
    is_speaking: bool = False
    current_language: str = "fr"
    user_name: str = "Fusion"

    # Event loop reference if needed
    loop: asyncio.AbstractEventLoop = field(default_factory=asyncio.get_event_loop)

    # Queues for inter-module communication
    audio_output_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    def set_state(self, new_state: JarvisState):
        self.state = new_state
