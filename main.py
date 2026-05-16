import sys
import asyncio
import logging
import traceback
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThread

from dotenv import load_dotenv
from jarvis.core.orchestrator import Jarvis
from jarvis.ui.hud import ArcReactor, JarvisSignals

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("Main")

class JarvisWorker(QThread):
    def __init__(self, signals):
        super().__init__()
        self.signals = signals

    def run(self):
        try:
            log.info("🧵 Worker thread démarré")
            asyncio.run(self.run_async())
        except Exception as e:
            log.error(f"❌ CRASH dans le worker thread: {e}")
            log.error(traceback.format_exc())

    async def run_async(self):
        try:
            log.info("🤖 Initialisation de Jarvis...")
            self.jarvis = Jarvis(signals=self.signals)
            log.info("🤖 Jarvis initialisé, démarrage...")
            await self.jarvis.run()
        except Exception as e:
            log.error(f"❌ Erreur dans le worker Jarvis: {e}")
            log.error(traceback.format_exc())

def main():
    log.info("🎬 Démarrage de JARVIS...")
    app = QApplication(sys.argv)

    log.info("🎨 Création du HUD Arc Reactor...")
    reactor = ArcReactor()
    screen = app.primaryScreen().availableGeometry()
    reactor.move(screen.center().x() - reactor.width() // 2,
                 screen.center().y() - reactor.height() // 2)
    reactor.show()
    log.info("✅ HUD affichée")

    log.info("📡 Configuration des signaux...")
    signals = JarvisSignals()
    signals.transcription_received.connect(reactor.set_text)
    signals.thinking_state_changed.connect(reactor.set_thinking)

    log.info("🧵 Démarrage du worker thread...")
    worker = JarvisWorker(signals)
    worker.start()

    try:
        log.info("🚀 Boucle Qt démarrée")
        sys.exit(app.exec())
    except KeyboardInterrupt:
        log.info("⏹️ Arrêt demandé par l'utilisateur")
        pass
    except Exception as e:
        log.error(f"❌ Erreur dans main: {e}")
        log.error(traceback.format_exc())

if __name__ == "__main__":
    main()
