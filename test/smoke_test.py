import asyncio
import os
import sys
import logging

# Ensure we can import jarvis
sys.path.append(os.getcwd())

from jarvis.perception.audio import VoiceActivityDetector
from jarvis.perception.stt import SpeechToText
from jarvis.tts.engine import TextToSpeech
from jarvis.core.context import JarvisContext

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("SmokeTest")

async def smoke_test():
    log.info("🚀 Démarrage du Smoke Test...")

    # 1. Test VAD Initialization
    try:
        log.info("🧪 Test Initialisation VAD...")
        vad = VoiceActivityDetector()
        log.info("✅ VAD OK.")
    except Exception as e:
        log.error(f"❌ VAD Échec: {e}")
        return False

    # 2. Test STT Initialization
    try:
        log.info("🧪 Test Initialisation STT (Whisper)...")
        stt = SpeechToText()
        log.info("✅ STT OK.")
    except Exception as e:
        log.error(f"❌ STT Échec: {e}")
        return False

    # 3. Test TTS Initialization
    try:
        log.info("🧪 Test Initialisation TTS (Kokoro)...")
        tts = TextToSpeech()
        if not tts.available:
            log.error("❌ TTS non disponible (assets manquants ?)")
            return False
        log.info("✅ TTS OK.")
    except Exception as e:
        log.error(f"❌ TTS Échec: {e}")
        return False

    log.info("✨ Tous les composants critiques ont été initialisés avec succès !")
    return True

if __name__ == "__main__":
    success = asyncio.run(smoke_test())
    sys.exit(0 if success else 1)
