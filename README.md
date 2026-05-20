# JARVIS - Local Voice Assistant

JARVIS is a production-grade, 100% local, and privacy-focused voice assistant. It is optimized for Windows environments with NVIDIA GPU acceleration (CUDA), providing a low-latency, "Iron Man"-like experience with a stylized HUD.

## 🚀 Key Features

- **100% Local & Private**: No data leaves your machine. All AI processing (STT, LLM, TTS) happens locally.
- **Real-time Perception**: Uses **Faster-Whisper** for high-accuracy transcription and **Silero VAD** for robust voice activity detection.
- **Smart Intelligence**: Integrated with **Ollama** (defaults to Llama 3) for natural conversations.
- **Human-like Speech**: Powered by **Kokoro-ONNX** for high-quality, low-latency text-to-speech synthesis.
- **Persistent Memory**: Remembers user facts and history using an optimized **SQLite FTS5** database.
- **Action System**: Can execute system commands, open applications, search the web, control Spotify, and integrate with Home Assistant.
- **Barge-in Support**: You can interrupt JARVIS while it's speaking, just like a real conversation.
- **Stylized HUD**: A unique "Arc Reactor" interface built with PyQt6.

## 🏗️ Architecture

JARVIS follows a modular, asynchronous architecture managed by a central **Orchestrator**:

- **Core**: Manages state transitions, inter-module communication, and audio buffering.
- **Perception**: Handles microphone input, Automatic Gain Control (AGC), and Speech-to-Text.
- **Intelligence**: Manages LLM streaming and long-term memory retrieval/storage.
- **TTS**: Handles real-time speech generation and playback synchronization.
- **Actions**: Parses and executes commands extracted from LLM responses.

## 🛠️ Requirements

- **Operating System**: Windows 10/11 (highly recommended for full feature support).
- **Python**: 3.10 or higher.
- **Hardware**:
    - Recommended: NVIDIA GPU with 8GB+ VRAM (for Faster-Whisper Medium and Llama 3 8B).
    - Minimum: 16GB RAM for CPU-only mode.
- **Software**: [Ollama](https://ollama.com/) must be installed and running.

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd jarvis-local
   ```

2. **Run the automated setup**:
   Double-click `setup_jarvis.bat`. This script will:
   - Create a virtual environment (`.venv`).
   - Detect your hardware (CPU vs. NVIDIA GPU).
   - Install the appropriate dependencies (Core + AI Stack).
   - Download the necessary local models (VAD, Kokoro).

3. **Ensure Ollama is ready**:
   Make sure Ollama is running and you have the default model pulled:
   ```bash
   ollama pull llama3:8b
   ```

## 🖥️ Usage

To launch the assistant, run the following command:
```bash
.venv\Scripts\python.exe main.py
```

### Testing your setup
You can verify that the critical AI components are correctly initialized by running the smoke test:
```bash
.venv\Scripts\python.exe test/smoke_test.py
```

## 📝 Configuration

System settings such as model sizes, voice selection, and application whitelists can be adjusted in `jarvis/utils/config.py`.

## 🛡️ License

This project is for educational and personal use. Refer to the individual licenses of the models used (Whisper, Llama, Kokoro) for more details.
