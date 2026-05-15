from enum import Enum, auto

class JarvisState(Enum):
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    EXECUTING = auto()
    SPEAKING = auto()
    BLOCKED = auto()
