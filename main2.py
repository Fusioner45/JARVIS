# main.py
# --------------------------------------------------------------
# JARVIS – version optimisée (QThread worker, filtrage, whitelist,
#        optimisation VRAM, protection audio)
# --------------------------------------------------------------
import sys
import logging
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget

# --- IMPORTS PROJET ------------------------------------------------
# Ajustez ces imports selon votre arborescence réelle
from audio_controller import AudioController          # éventuellement lève une exception au init
from stt import WhisperSTT                           # wrapper Whisper (faster‑whisper ou whisper.cpp)
from llm import OllamaLLM                            # wrapper Ollama
from tts import PiperTTS                             # ou tout autre moteur TTS
from command_executor import (                         # fonctions implémentant les actions autorisées
    open_app,
    get_gpu_temp,
    volume_control,
    play_music,
)

# --------------------------------------------------------------
# Configuration du logging (utile en mode silencieux)
# --------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Jarvis")

# --------------------------------------------------------------
# Whitelist des actions autorisées
# --------------------------------------------------------------
ALLOWED_ACTIONS = ["OPEN_APP", "GET_GPU_TEMP", "VOLUME_CONTROL", "PLAY_MUSIC"]

# --------------------------------------------------------------
# Phrases de rejet pour filtrer les hallucinations de Whisper
# --------------------------------------------------------------
REJECT_PHRASES = [
    "Amara.org",
    "Merci d'avoir regardé",
    # ajoutez d'autres chaînes si besoin
]

# --------------------------------------------------------------
# Worker QThread : toute la chaîne de traitement
# --------------------------------------------------------------
class JarvisWorker(QThread):
    """
    Thread de travail qui :
      1. Capture l'audio du microphone
      2. Transcrit avec Whisper
      3. Filtre les hallucinations
      4. Envoie le texte à Ollama (avec paramètres VRAM optimisés)
      5. Extrait la commande JSON de la réponse
      6. Vérifie la whitelist
      7. Exécute l'action via les fonctions de command_executor
      8. Renvoie le résultat (ou une erreur) via des signaux Qt
    """

    # Signaux destinés à l'UI (thread principal)
    status_changed = pyqtSignal(str)          # message d'état à afficher
    user_said = pyqtSignal(str)               # ce que l'utilisateur a dit (post‑STT)
    assistant_reply = pyqtSignal(str)         # texte à lire par TTS
    error_occurred = pyqtSignal(str)          # erreur à logger silencieusement

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._running = True

        # --- Initialisation des composants (avec protection) ---
        try:
            self.audio = AudioController()
            logger.info("AudioController initialisé avec succès.")
        except Exception as e:
            logger.error(f"Échec d'initialisation de AudioController : {e}")
            self.audio = None  # on continuera sans audio (les signaux d'erreur seront levés)

        try:
            self.stt = WhisperSTT(model_name_or_path="base")  # adaptez selon votre modèle
            logger.info("WhisperSTT prêt.")
        except Exception as e:
            logger.error(f"Échec d'initialisation de WhisperSTT : {e}")
            self.stt = None

        try:
            # Ollama : quantisation q4_0, contexte réduit, prédiction limitée
            self.llm = OllamaLLM(
                model="llama3",                     # ou le modèle que vous utilisez
                cache_type="q4_0",
                num_ctx=2048,
                num_predict=128,
                temperature=0.2,
            )
            logger.info("OllamaLLM configuré (q4_0, ctx=2048, pred=128).")
        except Exception as e:
            logger.error(f"Échec d'initialisation de OllamaLLM : {e}")
            self.llm = None

        try:
            self.tts = PiperTTS()  # ou autre moteur TTS
            logger.info("TTS initialisé.")
        except Exception as e:
            logger.error(f"Échec d'initialisation de TTS : {e}")
            self.tts = None

    # ----------------------------------------------------------
    # Boucle principale du thread
    # ----------------------------------------------------------
    def run(self):
        logger.info("JarvisWorker démarré.")
        while self._running:
            try:
                self.status_changed.emit("Écoute…")
                # 1️⃣ Capture audio (blocante mais dans le worker)
                audio_data = self.audio.record_chunk(duration=3.0) if self.audio else None
                if audio_data is None:
                    self.error_occurred.emit("AudioController indisponible.")
                    self.msleep(500)
                    continue

                # 2️⃣ Transcription Whisper
                if not self.stt:
                    self.error_occurred.emit("STT indisponible.")
                    self.msleep(500)
                    continue

                raw_text = self.stt.transcribe(audio_data).strip()
                if not raw_text:
                    # Silence pur → on ignore simplement
                    self.msleep(100)
                    continue

                logger.info(f"Whisper a retourné : '{raw_text}'")
                self.user_said.emit(raw_text)

                # 3️⃣ Filtrage des hallucinations (phrases de rejet)
                if any(rej.lower() in raw_text.lower() for rej in REJECT_PHRASES):
                    logger.info(f"Hallucination détectée («{raw_text}») → rejet.")
                    self.status_changed.emit("Hallucination filtrée.")
                    continue

                # 4️⃣ Envoi à Ollama (LLM)
                if not self.llm:
                    self.error_occurred.emit("LLM indisponible.")
                    self.msleep(500)
                    continue

                # Prompt simple : on demande à l'IA de répondre au format JSON
                # {"action": "...", "params": {...}}
                prompt = (
                    "Tu es un assistant vocal. À partir de la phrase suivante, "
                    "produis UNIQUEMENT un objet JSON valide contenant les champs "
                    '"action" (une des valeurs : OPEN_APP, GET_GPU_TEMP, VOLUME_CONTROL, PLAY_MUSIC) '
                    'et "params" (objet selon l\'action). '
                    f"Phrase utilisateur : \"{raw_text}\"\n"
                    "JSON :"
                )
                llm_response = self.llm.generate(prompt).strip()
                logger.debug(f"Réponse brute Ollama : {llm_response}")

                # 5️⃣ Extraction JSON (tolérance légère)
                import json, re
                json_match = re.search(r"\{.*\}", llm_response, re.DOTALL)
                if not json_match:
                    logger.warning("Aucun JSON détecté dans la réponse Ollama.")
                    self.error_occurred.emit("Réponse LLM invalide (pas de JSON).")
                    continue
                try:
                    command_obj = json.loads(json_match.group(0))
                except json.JSONDecodeError as je:
                    logger.error(f"JSON invalide : {je}")
                    self.error_occurred.emit("Réponse LLM invalide (JSON mal formé).")
                    continue

                action = command_obj.get("action", "").upper()
                params = command_obj.get("params", {})

                # 6️⃣ Vérification whitelist
                if action not in ALLOWED_ACTIONS:
                    logger.info(f"Action non autorisée reçue : {action}")
                    self.error_occurred.emit(f"Action refusée : {action}")
                    continue

                # 7️⃣ Exécution de l'action
                self.status_changed.emit(f"Exécution de {action}…")
                try:
                    if action == "OPEN_APP":
                        result = open_app(**params)
                    elif action == "GET_GPU_TEMP":
                        result = get_gpu_temp(**params)
                    elif action == "VOLUME_CONTROL":
                        result = volume_control(**params)
                    elif action == "PLAY_MUSIC":
                        result = play_music(**params)
                    else:
                        # Ne devrait jamais arriver grâce au whitelist
                        raise ValueError(f"Action inconnue : {action}")

                    logger.info(f"Action {action} exécutée avec résultat : {result}")
                    # Optionnel : faire parler le résultat via TTS
                    if self.tts and isinstance(result, str) and result:
                        self.tts.speak(result)
                        self.assistant_reply.emit(result)
                except Exception as exc:
                    logger.exception(f"Erreur lors de l'exécution de {action}")
                    self.error_occurred.emit(f"Erreur d'exécution : {exc}")

                # Petite pause avant la prochaine écoute (évite de saturer le CPU)
                self.msleep(300)

            except Exception as e:
                logger.exception("Exception inattendue dans la boucle du worker")
                self.error_occurred.emit(f"Exception worker : {e}")
                self.msleep(1000)

        logger.info("JarvisWorker arrêté.")

    # ----------------------------------------------------------
    # Méthode d'arrêt propre
    # ----------------------------------------------------------
    def stop(self):
        self._running = False
        self.wait()  # attend la fin du thread


# --------------------------------------------------------------
# Fenêtre principale minimale (PyQt6) – à adapter à votre UI réelle
# --------------------------------------------------------------
class JarvisMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JARVIS – Optimisé")
        self.resize(400, 200)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.lbl_status = QLabel("Prêt…")
        self.lbl_user = QLabel("")
        self.lbl_assistant = QLabel("")

        layout.addWidget(self.lbl_status)
        layout.addWidget(QLabel("Vous avez dit :"))
        layout.addWidget(self.lbl_user)
        layout.addWidget(QLabel("JARVIS répond :"))
        layout.addWidget(self.lbl_assistant)

        # Création et lancement du worker
        self.worker = JarvisWorker()
        self.worker.status_changed.connect(self.on_status_changed)
        self.worker.user_said.connect(self.on_user_said)
        self.worker.assistant_reply.connect(self.on_assistant_reply)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()

    # ---- Slots rattachés aux signaux du worker ----
    @pyqtSlot(str)
    def on_status_changed(self, msg: str):
        self.lbl_status.setText(msg)

    @pyqtSlot(str)
    def on_user_said(self, txt: str):
        self.lbl_user.setText(txt)

    @pyqtSlot(str)
    def on_assistant_reply(self, txt: str):
        self.lbl_assistant.setText(txt)

    @pyqtSlot(str)
    def on_error(self, err: str):
        # Log uniquement – ne pop‑up pas pour ne pas déranger l'utilisateur
        logger.warning(f"[UI] Erreur signalée : {err}")

    def closeEvent(self, event):
        """Arrêt propre du worker à la fermeture de la fenêtre."""
        self.worker.stop()
        super().closeEvent(event)


# --------------------------------------------------------------
# Point d'entrée
# --------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    win = JarvisMainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
