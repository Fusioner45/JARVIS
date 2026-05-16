import os
import logging

# Audio Settings
SAMPLE_RATE = 16000
FRAME_MS = 32
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)
SILENCE_FRAMES_THRESHOLD = 15

# STT Settings
WHISPER_MODEL_SIZE = "medium"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16" # Plus stable que int8_float16 sur de nombreux GPUs

# LLM Settings
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3:8b"

# TTS Settings
EDGE_TTS_VOICE = "fr-FR-DeniseNeural"

# Paths
USER_PROFILE = os.environ.get('USERPROFILE', r'C:\Users\Fusion')
DB_PATH = "memory.db"

# Security
APP_WHITELIST = {
    "vscode": os.path.join(USER_PROFILE, r"AppData\Local\Programs\Microsoft VS Code\Code.exe"),
    "spotify": os.path.join(USER_PROFILE, r"AppData\Roaming\Spotify\Spotify.exe"),
    "discord": os.path.join(USER_PROFILE, r"AppData\Local\Discord\Update.exe"),
    "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "calculatrice": "calc.exe",
}

# Playlists
PLAYLISTS = {
    "liké": "https://open.spotify.com/collection/tracks",
    "triste": "https://open.spotify.com/playlist/4WU64ygsmIDYDwPEI2BDnQ"
}
