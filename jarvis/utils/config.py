import os
import logging

# Audio Settings
SAMPLE_RATE = 16000
FRAME_MS = 32
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)
SILENCE_FRAMES_THRESHOLD = 15
VAD_THRESHOLD = 0.4
AGC_TARGET_RMS = 0.10
AGC_MAX_GAIN = 8.0

# STT Settings (Restored to High Quality CUDA)
WHISPER_MODEL_SIZE = "medium"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

# LLM Settings
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3:8b"

# TTS Settings
KOKORO_VOICE = "af_sky" # Default high quality voice
KOKORO_MODEL_PATH = "models/kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = "models/voices-v1.0.bin"

# Paths
USER_PROFILE = os.path.expanduser('~')
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
