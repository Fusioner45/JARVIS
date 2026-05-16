import sys
import asyncio
import logging
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

class JarvisWorker(QThread):
    def __init__(self, signals):
        super().__init__()
        self.signals = signals

    def run(self):
        try:
            asyncio.run(self.run_async())
        except Exception as e:
            logging.error(f"❌ Erreur critique asynchrone : {e}")

    async def run_async(self):
        try:
            self.jarvis = Jarvis(signals=self.signals)
            await self.jarvis.run()
        except Exception as e:
            logging.error(f"❌ Erreur fatale dans l'orchestrateur Jarvis : {e}")
            import traceback
            logging.error(traceback.format_exc())

def main():
    app = QApplication(sys.argv)
    logging.info("🚀 Lancement du HUD PyQt6...")

    try:
        reactor = ArcReactor()
        screen = app.primaryScreen().availableGeometry()
        reactor.move(screen.center().x() - reactor.width() // 2,
                     screen.center().y() - reactor.height() // 2)
        reactor.show()

        signals = JarvisSignals()
        signals.transcription_received.connect(reactor.set_text)
        signals.thinking_state_changed.connect(reactor.set_thinking)

        logging.info("🧠 Initialisation du worker JARVIS...")
        worker = JarvisWorker(signals)
        worker.start()

        exit_code = app.exec()
        logging.info(f"🏁 L'application s'est terminee avec le code : {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        logging.critical(f"💥 Crash de l'application : {e}")
        import traceback
        logging.critical(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
