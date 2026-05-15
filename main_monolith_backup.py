#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Jarvis V2 – Assistant vocal ultra‑rapide (Windows 11, RTX 3070 Ti)

Architecture :
    ├─ STT  : faster‑whisper (small) + Silero VAD  → CUDA
    ├─ LLM  : Ollama (llama3:8b ou phi3)                → HTTP localhost:11434/v1
    └─ TTS  : edge‑tts (voix FR) → MP3 → PCM via pydub/ffmpeg

Tout est asynchrone grâce à asyncio afin que l'écoute,
la réflexion et la parole puissent se chevaucher.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import asyncio
import logging
import sys
import io
import json
import re
import sqlite3
import threading
import time
import subprocess
import queue
from collections import deque
from typing import AsyncGenerator, List, Dict, Any

import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QGraphicsDropShadowEffect
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QPropertyAnimation, QEasingCurve, QPointF, QRectF
from PyQt6.QtGui import QColor, QPalette, QFont, QPainter, QRadialGradient, QPen, QBrush
import webbrowser
import os
import pyautogui
import sounddevice as sd
import torch
import pvporcupine
from pvrecorder import PvRecorder
from dotenv import load_dotenv
import pynvml
import pygetwindow as gw
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from faster_whisper import WhisperModel
import aiohttp
import edge_tts
from pydub import AudioSegment  # décodage MP3 → PCM
import pypdf

# ----------------------------------------------------------------------
# Persistence: SQLite Memory Core
# ----------------------------------------------------------------------
class JarvisMemory:
    def __init__(self, db_path="memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # Activation FTS5 pour recherche plein-texte performante
            # On utilise tokenize='porter' pour le stemming (plus intelligent)
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(fact_type, content, timestamp, expires_at, tokenize='porter')")
            conn.commit()

    def cleanup_obsolete(self):
        """Supprime les faits expirés."""
        with sqlite3.connect(self.db_path) as conn:
            now = time.time()
            conn.execute("DELETE FROM memory_fts WHERE expires_at IS NOT NULL AND expires_at < ?", (now,))
            conn.commit()

    def save_memory(self, fact_type: str, content: str, ttl_days: int = None):
        expires_at = time.time() + (ttl_days * 86400) if ttl_days else None
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO memory_fts (fact_type, content, timestamp, expires_at) VALUES (?, ?, ?, ?)",
                         (fact_type, content, time.time(), expires_at))
            conn.commit()
            log.info(f"🧠 Mémoire FTS5 : [{fact_type}] {content}")

    def save_task(self, task: str, due_time: str = None):
        # Les tâches expirent par défaut après 7 jours si non précisé
        self.save_memory("tâche", f"{task} (Due: {due_time})" if due_time else task, ttl_days=7)

    def index_pdf(self, pdf_path: str):
        """Extrait le texte d'un PDF et le stocke comme 'cours'."""
        try:
            with open(pdf_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"

                # On stocke par morceaux pour ne pas saturer une ligne
                chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                with sqlite3.connect(self.db_path) as conn:
                    file_name = os.path.basename(pdf_path)
                    for i, chunk in enumerate(chunks):
                        conn.execute("INSERT INTO memory_fts (fact_type, content, timestamp) VALUES (?, ?, ?)",
                                     (f"cours:{file_name}", chunk, time.time()))
                    conn.commit()
            log.info(f"📚 PDF indexé : {pdf_path}")
        except Exception as e:
            log.error(f"Erreur indexation PDF {pdf_path}: {e}")

    def query_memory(self, search_term: str) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            # Recherche FTS5 optimisée
            cursor = conn.execute("SELECT content FROM memory_fts WHERE content MATCH ? OR fact_type MATCH ?",
                                 (search_term, search_term))
            return [row[0] for row in cursor.fetchall()]

    def delete_memory(self, search_term: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM memory_fts WHERE content MATCH ? OR fact_type MATCH ?",
                         (search_term, search_term))
            conn.commit()
            log.info(f"🗑️ Mémoire supprimée : {search_term}")

    def get_all_context(self) -> str:
        """Récupère un résumé de tous les faits pertinents pour le prompt système."""
        with sqlite3.connect(self.db_path) as conn:
            self.cleanup_obsolete()
            # Priorité aux informations récentes et types importants
            cursor = conn.execute("SELECT fact_type, content FROM memory_fts ORDER BY timestamp DESC LIMIT 15")
            facts = [f"- {ft}: {c}" for ft, c in cursor.fetchall()]
            return "\n".join(facts) if facts else "Mémoire vide."

    def summarize_context(self, history: List[Dict]):
        """Placeholder pour une future fonction de résumé du contexte conversationnel."""
        # En production, on pourrait demander au LLM de résumer les 50 derniers messages
        pass

    def get_user_name(self) -> str:
        """Cherche le nom de l'utilisateur dans la mémoire."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT content FROM memory_fts WHERE fact_type = 'nom_utilisateur' LIMIT 1")
            row = cursor.fetchone()
            return row[0] if row else "Fusion"

# ----------------------------------------------------------------------
# Audio / Playlists
# ----------------------------------------------------------------------
PLAYLISTS = {
    "liké": "https://open.spotify.com/collection/tracks",
    "triste": "https://open.spotify.com/playlist/4WU64ygsmIDYDwPEI2BDnQ"
}

# ----------------------------------------------------------------------
# Command Mappings & Allowed Tags (Security & Mapping)
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Command Configuration & Security Whitelist
# ----------------------------------------------------------------------
# Mapping dynamique des applications via variables d'environnement
USER_PROFILE = os.environ.get('USERPROFILE', r'C:\Users\Fusion')
APP_WHITELIST = {
    "vscode": os.path.join(USER_PROFILE, r"AppData\Local\Programs\Microsoft VS Code\Code.exe"),
    "spotify": os.path.join(USER_PROFILE, r"AppData\Roaming\Spotify\Spotify.exe"),
    "discord": os.path.join(USER_PROFILE, r"AppData\Local\Discord\Update.exe"),
    "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "calculatrice": "calc.exe",
}

# ----------------------------------------------------------------------
# Configuration (à adapter si besoin)
# ----------------------------------------------------------------------
SAMPLE_RATE = 16000               # Hz – required by webrtcvad & faster‑whisper
FRAME_MS = 32                     # ms – size of each audio frame for VAD
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)  # samples per frame
# VAD_MODE deleted                      # 0‑3, 2 = bonne compromis sensibilité/robustesse
SILENCE_FRAMES_THRESHOLD = 15     # nombre de frames silencieuses pour finir une utterance
WHISPER_MODEL_SIZE = "medium"      # tiny, base, small, medium, large‑v2 …
WHISPER_DEVICE = "cuda"           # on utilise le GPU
WHISPER_COMPUTE_TYPE = "int8_float16"  # optimum pour RTX 30xx
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3:8b"        # ou "phi3" selon ce que vous avez installé
EDGE_TTS_VOICE = "fr-FR-DeniseNeural"  # voix française naturelle (Microsoft)

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("JarvisV2")

# ----------------------------------------------------------------------
# Helper: audio capture → async generator of raw PCM frames
# ----------------------------------------------------------------------
async def audio_frame_generator() -> AsyncGenerator[bytes, None]:
    """
    Capture audio depuis le microphone default et yield des frames PCM16
    de taille FRAME_SIZE (en octets). Fonctionne en boucle infinie.
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[bytes | None] = asyncio.Queue()

    def callback(indata, frames, time, status):
        """sounddevice callback (called from audio thread)."""
        if status:
            log.warning(f"Audio stream status: {status}")
        # Convert to mono int16 PCM
        mono = indata[:, 0] if indata.ndim > 1 else indata
        pcm16 = (mono * 32767).astype(np.int16).tobytes()
        # Put into queue (thread‑safe via call_soon_threadsafe)
        loop.call_soon_threadsafe(q.put_nowait, pcm16)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SIZE,
        callback=callback,
    ):
        log.info("🎤 Microphone ouvert – écoute en continu")
        while True:
            frame = await q.get()
            if frame is None:          # sentinel for shutdown
                break
            yield frame


# ----------------------------------------------------------------------
# VAD wrapper – Silero VAD (robust & accurate)
# ----------------------------------------------------------------------
class VoiceActivityDetector:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        log.info("Chargement de Silero VAD...")
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=True
        )
        self.sample_rate = sample_rate

    def is_speech(self, frame: bytes, threshold: float = 0.5, proximity_threshold: float = 0.01) -> bool:
        """
        Retourne True si la frame contient de la voix et respecte le seuil de proximité (gain).
        """
        audio_int16 = np.frombuffer(frame, dtype=np.int16)

        # Détection de proximité via RMS (énergie audio)
        rms = np.sqrt(np.mean(audio_int16.astype(np.float32)**2)) / 32768.0
        if rms < proximity_threshold:
            return False

        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        tensor_input = torch.from_numpy(audio_float32).unsqueeze(0) # Ajout de la dimension batch

        with torch.no_grad():
            confidence = self.model(tensor_input, self.sample_rate).item()
        return confidence > threshold


# ----------------------------------------------------------------------
# STT – faster‑whisper on CUDA
# ----------------------------------------------------------------------
class SpeechToText:
    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
    ):
        log.info(f"Chargement du modèle Whisper '{model_size}' sur {device} …")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        log.info("✅ Modèle Whisper chargé.")

    async def transcribe(self, audio_frames: List[bytes], language: str = "fr") -> str:
        """
        Transcrit une liste de frames PCM16 en texte avec support multi-langue.
        Fonctionne en thread séparé pour ne pas bloquer la boucle asyncio.
        """
        loop = asyncio.get_running_loop()

        def _sync_transcribe():
            # Concaténer toutes les frames en un seul np.ndarray float32 [-1,1]
            audio_np = np.frombuffer(b"".join(audio_frames), dtype=np.int16).astype(np.float32) / 32768.0
            # faster‑whisper attend un tableau 1‑D
            segments, info = self.model.transcribe(
                audio_np,
                language=language,
                task="transcribe",
                initial_prompt="Ceci est une conversation en français uniquement. L'utilisateur parle de ses cours, de ses projets à Issoire et de musique. Ne jamais traduire en anglais.",
                beam_size=5,
                best_of=5,
                suppress_tokens=[-1],
                vad_filter=False,  # on fait notre propre VAD en amont
            )
            text = " ".join(seg.text for seg in segments).strip()
            return text

        try:
            return await loop.run_in_executor(None, _sync_transcribe)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                log.error("💥 GPU saturé lors de la transcription – vidage du cache.")
                import torch

                torch.cuda.empty_cache()
                log.info("🔄 Basculement temporaire sur CPU pour cette utterance.")
                self.model = WhisperModel(
                    WHISPER_MODEL_SIZE, device="cpu", compute_type="int8"
                )
                return await loop.run_in_executor(None, _sync_transcribe)
            else:
                raise


# ----------------------------------------------------------------------
# LLM – appel à Ollama (API compatible OpenAI)
# ----------------------------------------------------------------------
class LlmClient:
    def __init__(self, base_url: str = OLLAMA_HOST, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session: aiohttp.ClientSession | None = None

    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    async def generate_stream(self, messages: List[dict]) -> AsyncGenerator[str, None]:
        """
        Envoie les messages à Ollama et yield les morceaux de texte au fur et à mesure.
        Utilise l'endpoint /v1/chat/completions avec stream=True.
        Inclut une logique de retry pour gérer les erreurs 500.
        """
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 512,
            "stream": True,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                await self._ensure_session()
                # On utilise un timeout plus long pour la connexion et on laisse le stream s'écouler
                async with self.session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(connect=5, total=120)
                ) as resp:
                    if resp.status == 500:
                        log.warning(f"Ollama error 500 (attempt {attempt+1}/{max_retries}). Retrying...")
                        await asyncio.sleep(1)
                        continue

                    resp.raise_for_status()
                    # Lecture du stream JSON-L (OpenAI format)
                    async for line in resp.content:
                        if not line:
                            continue
                        line_str = line.decode("utf-8").strip()
                        if line_str.startswith("data: "):
                            data_content = line_str[6:]
                            if data_content == "[DONE]":
                                break
                            try:
                                data = json.loads(data_content)
                                token = data["choices"][0]["delta"].get("content", "")
                                if token:
                                    yield token
                            except Exception:
                                continue
                    return # Succès
            except Exception as e:
                if attempt == max_retries - 1:
                    log.error(f"Erreur finale lors du stream Ollama : {e}")
                    yield "Désolé, je n'ai pas pu obtenir de réponse."
                else:
                    log.warning(f"Ollama stream error: {e}. Retrying...")
                    await asyncio.sleep(1)


# ----------------------------------------------------------------------
# TTS – edge‑tts (voix FR) → MP3 → PCM via pydub/ffmpeg
# ----------------------------------------------------------------------
class TextToSpeech:
    def __init__(self, voice: str = EDGE_TTS_VOICE):
        self.voice = voice
        self.use_morphing = False

    async def speak(self, text: str, audio_queue: asyncio.Queue):
        """
        Lit le texte à voix haute grâce à edge‑tts avec gestion d'état is_speaking.
        """
        if not text:
            return

        Jarvis.is_speaking = True
        communicate = edge_tts.Communicate(text, voice=self.voice)
        mp3_data = io.BytesIO()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_data.write(chunk["data"])

        if mp3_data.tell() == 0:
            return

        mp3_data.seek(0)
        loop = asyncio.get_running_loop()

        def _decode():
            try:
                audio_segment = AudioSegment.from_file(mp3_data, format="mp3")
                audio_segment = audio_segment.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
                return np.frombuffer(audio_segment.raw_data, dtype=np.int16)
            except Exception as e:
                log.error(f"Erreur de décodage MP3 : {e}")
                return None

        pcm_array = await loop.run_in_executor(None, _decode)
        if pcm_array is not None:
            if self.use_morphing:
                pcm_array = await loop.run_in_executor(None, VoiceMorpher.apply_robot_filter, pcm_array, SAMPLE_RATE)
            await audio_queue.put(pcm_array)
            # On laisse is_speaking à True tant que la queue n'est pas vide (géré par le run)


# ----------------------------------------------------------------------
# UI PyQt6 - Cyberpunk HUD (Stark/Iron Man Style)
# ----------------------------------------------------------------------
class ArcReactor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(600, 700)

        # Glassmorphism
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(0, 20, 40, 0.4);
                border-radius: 30px;
                border: 2px solid rgba(0, 242, 255, 0.2);
            }
        """)

        # Layout
        layout = QVBoxLayout(self)
        layout.addSpacing(450)

        self.label = QLabel("SYSTEM ONLINE", self)
        # Typographie militaire
        font = QFont("OCR A Extended", 12)
        if font.family() == "OCR A Extended": pass
        else: font = QFont("Consolas", 12)

        self.label.setFont(font)
        self.label.setStyleSheet("color: #00f2ff; background: transparent; border: none; padding: 20px;")
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(self.label)

        # Animation parameters
        self.angle_outer = 0
        self.pulse_inner = 0
        self.is_thinking = False
        self.target_text = ""
        self.current_text = ""

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(20) # 50 FPS

    def animate(self):
        """Animation cycle with extreme safety checks."""
        try:
            if not hasattr(self, 'label') or self.label is None:
                return

            # Rotation ring
            self.angle_outer = (self.angle_outer + (4 if self.is_thinking else 1)) % 360
            self.pulse_inner += 0.15 if self.is_thinking else 0.05

            # Typewriter effect
            if hasattr(self, 'target_text') and self.target_text:
                if len(self.current_text) < len(self.target_text):
                    self.current_text += self.target_text[len(self.current_text)]
                    self.label.setText(self.current_text)

            self.update()
        except Exception as e:
            log.debug(f"UI Animate error suppressed: {e}")

    def paintEvent(self, event):
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            center = QPointF(self.width() / 2, 250)

            # 1. Global Neon Glow
            glow = QRadialGradient(center, 300)
            glow.setColorAt(0, QColor(0, 242, 255, 30))
            glow.setColorAt(1, Qt.GlobalColor.transparent)
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(center, 300, 300)

            # 2. Outer Ring (Rotating)
            painter.save()
            painter.translate(center)
            painter.rotate(self.angle_outer)

            pen_outer = QPen(QColor(0, 242, 255, 100), 2)
            pen_outer.setDashPattern([10, 10])
            painter.setPen(pen_outer)
            painter.drawEllipse(QRectF(-120, -120, 240, 240))

            # Outer thick segments
            painter.setPen(QPen(QColor(0, 242, 255, 180), 5))
            for i in range(0, 360, 60):
                painter.drawArc(QRectF(-125, -125, 250, 250), i * 16, 30 * 16)
            painter.restore()

            # 3. Inner Ring (Pulsing)
            inner_scale = 1.0 + 0.1 * np.sin(self.pulse_inner)
            inner_radius = 60 * inner_scale

            color_inner = QColor(0, 242, 255, 220) if not self.is_thinking else QColor(255, 50, 50, 220)
            painter.setPen(QPen(color_inner, 3))
            painter.drawEllipse(center, inner_radius, inner_radius)

            # Inner glow
            inner_glow = QRadialGradient(center, inner_radius)
            inner_glow.setColorAt(0, color_inner)
            inner_glow.setColorAt(1, Qt.GlobalColor.transparent)
            painter.setBrush(QBrush(inner_glow))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(center, inner_radius, inner_radius)

            # 4. Core Triangle/Circle
            painter.setPen(QPen(QColor(255, 255, 255, 200), 2))
            painter.drawEllipse(center, 15, 15)
        except Exception:
            # Sécurité massive pour éviter tout crash de l'UI
            pass

    def set_text(self, text):
        if text != self.target_text:
            self.target_text = text
            self.current_text = ""
            self.label.setText("")

    def set_thinking(self, thinking: bool):
        self.is_thinking = thinking


class JarvisSignals(QObject):
    transcription_received = pyqtSignal(str)
    thinking_state_changed = pyqtSignal(bool)


# ----------------------------------------------------------------------
# Helper: Voice Morphing (FFmpeg)
# ----------------------------------------------------------------------
class VoiceMorpher:
    @staticmethod
    def apply_robot_filter(input_pcm: np.ndarray, sample_rate: int) -> np.ndarray:
        """Applique un effet de voix robotique via FFmpeg."""
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fin, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fout:
            sf.write(fin.name, input_pcm, sample_rate)
            fin.close()
            fout.close()

            # Filtre FFmpeg : pitch shift down + vibrato (robotique)
            cmd = [
                'ffmpeg', '-y', '-i', fin.name,
                '-af', 'asetrate=16000*0.8,atempo=1.25,vibrato=f=10:d=0.5',
                fout.name
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STNULL)

            out_data, _ = sf.read(fout.name, dtype='int16')
            os.unlink(fin.name)
            os.unlink(fout.name)
            return out_data

# ----------------------------------------------------------------------
# Helper: Windows Audio Ducking
# ----------------------------------------------------------------------
class AudioController:
    def __init__(self):
        from ctypes import cast, POINTER
        try:
            devices = AudioUtilities.GetSpeakers()
            self.interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume = cast(self.interface, POINTER(IAudioEndpointVolume))
        except Exception as e:
            log.warning(f"AudioController (PyCAW) failed: {e}. Using PowerShell Fallback.")
            self.volume = None

    def set_ducking(self, duck: bool):
        if self.volume:
            try:
                target = 0.1 if duck else 1.0
                self.volume.SetMasterVolumeLevelScalar(target, None)
                return
            except Exception:
                pass

        # Fallback PowerShell (Slower but reliable)
        try:
            vol = 10 if duck else 100
            subprocess.run(["powershell", "-Command", f"(Get-WmiObject -Class Win32_AudioControl).SetVolume({vol})"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

# ----------------------------------------------------------------------
# Helper: Wake-Word Detection (Picovoice Porcupine)
# ----------------------------------------------------------------------
class WakeWordDetector:
    def __init__(self, access_key: str, keywords: List[str] = ["jarvis"]):
        self.porcupine = pvporcupine.create(access_key=access_key, keywords=keywords)
        self.recorder = PvRecorder(device_index=-1, frame_length=self.porcupine.frame_length)

    def listen(self):
        self.recorder.start()
        log.info("👂 En attente du mot-clé (Porcupine)...")
        try:
            while True:
                pcm = self.recorder.read()
                keyword_index = self.porcupine.process(pcm)
                if keyword_index >= 0:
                    log.info("✨ Mot-clé détecté !")
                    return True
        finally:
            self.recorder.stop()

# ----------------------------------------------------------------------
# Helper: Windows App Resolver
# ----------------------------------------------------------------------
class AppResolver:
    """Résout dynamiquement les chemins des applications Windows."""
    @staticmethod
    def find_app(app_name: str) -> str:
        # 1. Check Whitelist first
        if app_name in APP_WHITELIST:
            return APP_WHITELIST[app_name]

        # 2. Try simple command (for apps in PATH)
        try:
            full_path = subprocess.check_output(['where', app_name], stderr=subprocess.STNULL).decode().splitlines()[0]
            return full_path
        except: pass

        # 3. Registry Lookup (Example for VS Code)
        if "code" in app_name.lower() or "vscode" in app_name.lower():
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\Applications\Code.exe\shell\open\command")
                val, _ = winreg.QueryValueEx(key, "")
                return val.split('"')[1]
            except: pass

        return app_name # Fallback to name

# ----------------------------------------------------------------------
# Helper: Command Executor (Zero-Trust) & Guardrail (Anti-Loop)
# ----------------------------------------------------------------------
class ActionGuard:
    """Gère les cooldowns, l'historique et bloque les répétitions abusives."""
    def __init__(self, cooldown: float = 10.0):
        self.cooldown = cooldown
        self.history: List[Dict[str, Any]] = []
        self.last_execution: Dict[str, float] = {}
        self.failed_counts: Dict[str, int] = {}

    def is_blocked(self, cmd_id: str) -> bool:
        now = time.time()

        # 1. Backoff : Si une commande a échoué trop de fois
        if self.failed_counts.get(cmd_id, 0) >= 2:
            log.error(f"⛔ Commande bannie (trop d'échecs) : {cmd_id}")
            return True

        # 2. Cooldown Check
        if cmd_id in self.last_execution:
            if now - self.last_execution[cmd_id] < self.cooldown:
                log.warning(f"🚫 Action bloquée (Cooldown 10s) : {cmd_id}")
                return True

        # 3. Duplicate Loop Check (3 dernières actions identiques)
        if len(self.history) >= 3:
            recent_cmds = [h['cmd_id'] for h in self.history[-3:]]
            if all(c == cmd_id for c in recent_cmds):
                log.error(f"🛑 Boucle infinie détectée pour {cmd_id}.")
                return True

        self.last_execution[cmd_id] = now
        return False

    def record_result(self, cmd_id: str, status: str):
        self.history.append({"cmd_id": cmd_id, "status": status, "ts": time.time()})
        if "FAILED" in status:
            self.failed_counts[cmd_id] = self.failed_counts.get(cmd_id, 0) + 1
        else:
            self.failed_counts[cmd_id] = 0 # Reset si succès

class CommandExecutor:
    """Exécute des commandes système avec validation stricte (Zero-Trust)."""

    @staticmethod
    def execute(cmd_name: str, args: List[str]) -> str:
        forbidden = [';', '&', '|', '$', '>', '<', '`']
        for arg in args:
            if any(c in str(arg) for c in forbidden):
                return "FAILED: Injection character detected"

        try:
            if cmd_name == "OPEN_APP":
                app_id = args[0].lower()
                if app_id == "youtube":
                    webbrowser.open("https://youtube.com")
                    return "REAL_SUCCESS: YouTube opened"

                target = AppResolver.find_app(app_id)
                if os.path.exists(target) or target.endswith(".exe"):
                    subprocess.Popen([target], shell=False)
                    return f"REAL_SUCCESS: {app_id} launched"
                return f"FAILED: App {app_id} not found at {target}"

            elif cmd_name == "SEARCH_WEB":
                webbrowser.open(f"https://www.google.com/search?q={args[0]}")
                return "SUCCESS: Web search triggered"

            elif cmd_name == "START_FILE":
                path = args[0]
                if os.path.exists(path) and (path.startswith(r"C:") or path.startswith(os.environ.get('USERPROFILE', ''))):
                    os.startfile(path)
                    return f"SUCCESS: File {path} opened"
                return "FAILED: Path access denied or not found"

            return "FAILED: Unknown internal command"
        except Exception as e:
            return f"FAILED: {str(e)}"

# ----------------------------------------------------------------------
# Helper: Command Parser (Deterministic)
# ----------------------------------------------------------------------
class CommandParser:
    """Isole et nettoie les tags [CMD: ...] dans le flux du LLM."""

    @staticmethod
    def extract_all(text: str) -> List[str]:
        # Capture tout ce qui ressemble à [CMD: ...] même avec des retours à la ligne
        pattern = r"\[\s*CMD\s*:\s*(.*?)\s*\]"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        return [f"[CMD: {m.strip()}]" for m in matches]

    @staticmethod
    def parse_call(cmd_tag: str) -> tuple[str, List[str]]:
        """Extrait le nom et les arguments d'un tag nettoyé."""
        inner = cmd_tag.replace("[CMD:", "").replace("]", "").strip()
        if "(" not in inner:
            return inner, []

        name = inner.split("(")[0].strip()
        # Extraction intelligente des arguments séparés par des virgules
        raw_args = inner[len(name):].strip("() ")
        # Regex pour splitter par virgule SAUF si dans des quotes (simple mais efficace ici)
        args = [a.strip().strip("'\"") for a in re.split(r",(?=(?:[^']*'[^']*')*[^']*$)", raw_args)]
        return name, [a for a in args if a]

# ----------------------------------------------------------------------
# Helper: GPU Monitoring (NVML)
# ----------------------------------------------------------------------
class GpuMonitor:
    def __init__(self):
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0) # RTX 3070 Ti
            self.enabled = True
        except Exception:
            self.enabled = False

    def get_temperature(self) -> int:
        if self.enabled:
            return pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
        return -1

# ----------------------------------------------------------------------
# Agent Core: State Machine & Intent Classification
# ----------------------------------------------------------------------
class JarvisState:
    CHAT = "CHAT"
    ACTION = "ACTION"
    MEMORY = "MEMORY"
    BLOCKED = "BLOCKED"

class IntentClassifier:
    @staticmethod
    def classify(text: str) -> str:
        text = text.lower()
        # Mots-clés déclencheurs d'action
        action_keywords = ["ouvre", "lance", "cherche", "musique", "température", "mémorise", "rappelle"]
        if any(kw in text for kw in action_keywords):
            return JarvisState.ACTION
        return JarvisState.CHAT

# ----------------------------------------------------------------------
# Orchestrateur principal – boucle async
# ----------------------------------------------------------------------
class Jarvis:
    # État global partagé pour éviter le Larsen
    is_speaking = False

    def __init__(self, signals: JarvisSignals = None):
        self.signals = signals
        self.memory = JarvisMemory()
        self.audio_ctrl = AudioController()
        self.gpu_mon = GpuMonitor()
        self.guard = ActionGuard()
        self.vad = VoiceActivityDetector()
        self.stt = SpeechToText()
        self.llm = LlmClient()
        self.tts = TextToSpeech()

        # Buffers
        self._pre_roll_buffer: deque[bytes] = deque(maxlen=10) # ~320ms de "passé"
        self._speech_buffer: List[bytes] = []
        self._silence_count = 0
        self._is_speaking = False

        # Audio output - Producteur-Consommateur (Glitch-Free)
        self.output_queue = queue.Queue(maxsize=100)

        # Initial context loading from memory
        context = self.memory.get_all_context()
        self.user_name = self.memory.get_user_name()
        self.current_language = "fr"

        self.history = [
            {"role": "system", "content": f"""Tu es JARVIS, un agent de type 'Chain-of-Thought'.
Ton objectif est l'action déterministe. Tu réponds en Français, de manière ultra-concise.

PROTOCOLE DE RÉPONSE :
1. PENSÉE : Analyse l'intention (Conversation ou Commande ?).
2. ACTION : Si une commande est requise, génère-la SYSTÉMATIQUEMENT au début.
3. VÉRIFICATION : Si une action a échoué précédemment, propose une alternative ou cherche sur le web.

CONSIGNES CRITIQUES :
- Ne propose JAMAIS d'actions si l'utilisateur pose une question simple (ex: "Tu peux parler ?").
- Si l'utilisateur mentionne une application, AGIS immédiatement sans demander de permission.
- Ne sois jamais bavard. Pas de politesse inutile, pas de disclaimer d'IA.
- Si l'utilisateur dit "Triste", lance la musique triste : [CMD: PLAY_MUSIC('triste')].
- Si l'utilisateur demande si tu peux parler, réponds simplement par l'affirmative, ne déclenche pas d'outil.
- Ton utilisateur est {self.user_name}.

CONTEXTE MÉMOIRE :
{context}

WHITELIST ACTIONS :
- [CMD: OPEN_APP('vscode'|'spotify'|'discord'|'youtube')]
- [CMD: SEARCH_WEB('requête')]
- [CMD: PLAY_MUSIC('mot-clé')]
- [CMD: SAVE_FACT('type', 'contenu')]
- [CMD: DELETE_FACT('terme')]
- [CMD: SAVE_TASK('tâche', 'échéance')]
- [CMD: GET_GPU_TEMP()]
- [CMD: SPLIT_SCREEN('app1', 'app2')]
- [CMD: WORK_MODE()]
- [CMD: SCREENSHOT_ANALYZE()]
- [CMD: HA_CONTROL('entité', 'service')]

Exemple : "C'est fait, {self.user_name}. [CMD: OPEN_APP('spotify')]"
"""}
        ]
        self.history_limit = 10

    async def _audio_listener(self) -> AsyncGenerator[str, None]:
        """
        Écoute en continu le microphone avec verrou anti-Larsen.
        """
        start_time = 0
        async for frame in audio_frame_generator():
            # ANTI-LARSEN : Si Jarvis parle, on jette l'audio
            if Jarvis.is_speaking:
                self._speech_buffer = []
                continue

            is_speech = self.vad.is_speech(frame)

            if is_speech:
                if not self._is_speaking:
                    start_time = time.time()
                if not self._is_speaking:
                    log.info("🎤 Début de parole détecté")
                    self._is_speaking = True

                    # Interruption (Barge-in) : si Jarvis parlait, on l'arrête
                    if hasattr(self, "_response_task") and not self._response_task.done():
                        log.info("🚫 Interruption (Barge-in) détectée – silence !")
                        self._response_task.cancel()

                    # On vide le buffer audio IMMÉDIATEMENT pour un silence instantané
                    self._clear_audio_buffer()
                    while not self.audio_queue.empty():
                        try:
                            self.audio_queue.get_nowait()
                            self.audio_queue.task_done()
                        except asyncio.QueueEmpty:
                            break

                    # On initialise le speech buffer avec le pre-roll pour ne pas couper le début
                    self._speech_buffer = list(self._pre_roll_buffer)

                self._speech_buffer.append(frame)
                self._silence_count = 0
            else:
                if self._is_speaking:
                    self._speech_buffer.append(frame)
                    self._silence_count += 1

                    if self._silence_count > SILENCE_FRAMES_THRESHOLD:
                        log.info("⏹️ Fin de parole détectée")
                        self._is_speaking = False
                        # Fin d'énoncé : on transmet le buffer
                        if self._speech_buffer:
                            duration = time.time() - start_time
                            self.audio_ctrl.set_ducking(True)
                            transcript = await self.stt.transcribe(self._speech_buffer, language=self.current_language)
                            self.audio_ctrl.set_ducking(False)

                            # WHISPER HALLUCINATION FILTER (Production Grade)
                            hallucinations = [
                                "Merci d'avoir regardé", "Bye", "C'est tout", "S'abonner", "vidéo",
                                "Mettez un pouce bleu", "Sous-titres", "Transcription", "J'avise que"
                            ]
                            # Filtre de durée, de mots-clés et d'entropie simple
                            if transcript:
                                clean_t = transcript.strip()
                                # 1. Exact match parasites
                                blacklist = ["Merci d'avoir regardé cette vidéo.", "Merci.", "Bye.", "C'est tout."]
                                if clean_t in blacklist:
                                    transcript = ""
                                # 2. Duration based filtering
                                elif duration < 1.5 and any(h.lower() in clean_t.lower() for h in hallucinations):
                                    log.info(f"🤫 Hallucination filtrée ({duration:.1f}s) : {transcript}")
                                    transcript = ""
                                # 3. Entropy check (too many repetitions)
                                elif len(set(clean_t.split())) < len(clean_t.split()) / 3 and len(clean_t.split()) > 5:
                                    log.warning(f"🌀 Entropie trop faible (Répétition détectée) : {transcript}")
                                    transcript = ""

                            if transcript:
                                log.info(f"🗣️ Transcription : {transcript}")
                            if self.signals:
                                self.signals.transcription_received.emit(transcript)
                                yield transcript

                        # Reset
                        self._speech_buffer = []
                        self._silence_count = 0
                else:
                    # On remplit le buffer de pre-roll quand on ne parle pas
                    self._pre_roll_buffer.append(frame)

    async def run(self):
        """
        Boucle principale avec Wake-word, streaming LLM et TTS phrase par phrase.
        """
        load_dotenv()
        access_key = os.getenv("PICOVOICE_ACCESS_KEY")

        # Initialisation du détecteur de mot-clé si la clé est présente
        self.ww_detector = None
        if access_key:
            self.ww_detector = WakeWordDetector(access_key)

        log.info("🚀 Jarvis V4 Ultimate démarré !")

        # Salutation initiale
        greeting = f"Systèmes en ligne. Bonjour {self.user_name}."
        if self.signals:
            self.signals.transcription_received.emit(greeting)

        # File d'attente asynchrone (Production)
        self.audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

        # Stream sounddevice avec callback non-bloquant (Glitch-Free)
        def audio_callback(outdata, frames, time, status):
            if status:
                log.warning(f"Audio output status: {status}")

            try:
                # On essaie de récupérer un bloc prêt de la queue
                data = self.output_queue.get_nowait()
                outdata[:, 0] = data
            except queue.Empty:
                # Pas de données ? On remplit de silence (non-bloquant)
                outdata.fill(0)

        output_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            callback=audio_callback,
            blocksize=FRAME_SIZE,
        )
        output_stream.start()

        # Tâche de fond pour consommer la queue
        async def audio_worker():
            while True:
                chunk = await self.audio_queue.get()
                if chunk is None: break
                # On concatène au buffer interne
                self._append_audio_chunk(chunk)
                self.audio_queue.task_done()

                # S'il n'y a plus rien à dire, on libère le micro
                if self.audio_queue.empty():
                    Jarvis.is_speaking = False

        worker_task = asyncio.create_task(audio_worker())

        # Parler la salutation
        await self.tts.speak(greeting, self.audio_queue)

        try:
            while True:
                # Mode Wake-Word
                if self.ww_detector:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, self.ww_detector.listen)

                async for user_text in self._audio_listener():
                    # Une fois qu'on a un texte, on sort de l'écoute continue pour traiter (et revenir au wake-word après)

                    # Barge-in : annuler la réponse et vider l'audio
                    if hasattr(self, "_response_task") and not self._response_task.done():
                        log.info("🚫 Interruption (Barge-in) – on coupe la parole.")
                        self._response_task.cancel()

                    # Vider le buffer audio pour arrêter de parler immédiatement
                    self._clear_audio_buffer()
                    while not self.audio_queue.empty():
                        try:
                            self.audio_queue.get_nowait()
                            self.audio_queue.task_done()
                        except asyncio.QueueEmpty:
                            break

                    self._response_task = asyncio.create_task(self._process_and_respond(user_text))
                    await self._response_task

                    # Si on est en mode wake-word, on ne boucle qu'une fois par détection
                    if self.ww_detector:
                        break
        finally:
            output_stream.stop()
            output_stream.close()
            await self.audio_queue.put(None)
            await worker_task
            await self.llm.close()
            log.info("🔌 Session Ollama fermée.")

    def _append_audio_chunk(self, chunk):
        """Découpe et envoie les données dans la queue synchrone (Producteur)."""
        # S'assurer que le chunk est en int16 et mono
        if chunk is None or len(chunk) == 0: return

        # On découpe en blocs exacts de taille FRAME_SIZE pour le callback sounddevice
        for i in range(0, len(chunk), FRAME_SIZE):
            segment = chunk[i:i + FRAME_SIZE]
            if len(segment) < FRAME_SIZE:
                pad = np.zeros(FRAME_SIZE - len(segment), dtype=np.int16)
                segment = np.concatenate([segment, pad])

            try:
                self.output_queue.put(segment, block=False)
            except queue.Full:
                # En production, on pourrait attendre un peu ou logger
                break

    def _clear_audio_buffer(self):
        """Vide la file d'attente audio immédiatement."""
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break

    def _execute_command(self, cmd_tag: str) -> str:
        """Parse et exécute un bloc [CMD: ...] avec retour d'état et guardrail."""
        try:
            cmd_name, args = CommandParser.parse_call(cmd_tag)
            cmd_id = f"{cmd_name}:{args}"

            # 1. Action Guardrail (Anti-loop)
            if self.guard.is_blocked(cmd_id):
                return "BLOCKED: Action frequency too high or loop detected"

            # 2. Dispatching
            result = "SUCCESS: Action executed"

            if cmd_name == "OPEN_APP":
                result = CommandExecutor.execute("OPEN_APP", args)
                if "SUCCESS" in result and "spotify" in str(args).lower():
                    threading.Thread(target=lambda: (time.sleep(3), pyautogui.press('enter')), daemon=True).start()

            elif cmd_name == "SEARCH_WEB":
                result = CommandExecutor.execute("SEARCH_WEB", args)

            elif cmd_name == "PLAY_MUSIC":
                query = args[0].lower()
                target_url = None
                for keyword, url in PLAYLISTS.items():
                    if keyword in query:
                        target_url = url; break

                if not target_url and any(w in query for w in ["local", "mon pc"]):
                    result = CommandExecutor.execute("START_FILE", [r'C:\musique'])
                else:
                    url = target_url or f"https://www.youtube.com/results?search_query={args[0]}"
                    webbrowser.open(url)
                    threading.Thread(target=lambda: (time.sleep(5), pyautogui.press('space')), daemon=True).start()
                    result = f"SUCCESS: Music started on {url}"

            elif cmd_name == "SAVE_FACT":
                self.memory.save_memory(args[0], args[1])
            elif cmd_name == "DELETE_FACT":
                self.memory.delete_memory(args[0])
            elif cmd_name == "SAVE_TASK":
                self.memory.save_task(args[0], args[1] if len(args) > 1 else None)
            elif cmd_name == "SWITCH_LANG":
                self.current_language = args[0].lower()
            elif cmd_name == "GET_GPU_TEMP":
                temp = self.gpu_mon.get_temperature()
                result = f"SUCCESS: GPU Temperature is {temp}°C"
            elif cmd_name == "SPLIT_SCREEN":
                try:
                    windows = gw.getAllWindows()
                    w1 = [w for w in windows if args[0].lower() in w.title.lower()]
                    w2 = [w for w in windows if args[1].lower() in w.title.lower()]
                    if w1 and w2:
                        w1[0].restore(); w1[0].moveTo(0, 0); w1[0].resizeTo(960, 1080)
                        w2[0].restore(); w2[0].moveTo(960, 0); w2[0].resizeTo(960, 1080)
                        result = "SUCCESS: Windows split"
                    else: result = "FAILED: Windows not found"
                except Exception as e: result = f"FAILED: {e}"
            elif cmd_name == "WORK_MODE":
                CommandExecutor.execute("OPEN_APP", ["vscode"])
                self._execute_command("[CMD: PLAY_MUSIC('triste')]")
                result = "SUCCESS: Work mode activated"
            elif cmd_name == "GET_WEATHER":
                webbrowser.open(f"https://www.google.com/search?q=meteo+{args[0]}")
            elif cmd_name == "CALC_TRIP":
                webbrowser.open(f"https://www.google.com/maps/dir/{args[0]}/{args[1]}")
            elif cmd_name == "SET_VOICE_MORPH":
                self.tts.use_morphing = (args[0].lower() == "true")
            elif cmd_name == "INDEX_PDF":
                self.memory.index_pdf(args[0])
            elif cmd_name == "SCREENSHOT_ANALYZE":
                asyncio.create_task(self._screenshot_and_analyze())
            elif cmd_name == "HA_CONTROL":
                asyncio.create_task(self._ha_control(args[0], args[1]))

            log.info(f"➡️ Result : {result}")
            self.guard.record_result(cmd_id, result)
            return result

        except Exception as e:
            err = f"FAILED: {str(e)}"
            log.error(f"❌ Parser/Executor Error: {err}")
            return err

    async def _screenshot_and_analyze(self):
        """Prend une capture d'écran et l'analyse via Ollama (Moondream)."""
        import base64
        try:
            screenshot = pyautogui.screenshot()
            img_byte_arr = io.BytesIO()
            screenshot.save(img_byte_arr, format='PNG')
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

            log.info("📸 Analyse de l'écran en cours...")
            url = f"{OLLAMA_HOST}/api/generate"
            payload = {
                "model": "moondream",
                "prompt": "Décris brièvement ce que tu vois sur cet écran.",
                "images": [img_base64],
                "stream": False
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        description = data.get("response", "Je n'ai pas pu analyser l'image.")
                        log.info(f"👁️ Vision : {description}")
                        await self.tts.speak(f"Sur votre écran, je vois : {description}", self.audio_queue)
                    else:
                        await self.tts.speak("Désolé Fusion, ma vision est temporairement indisponible.", self.audio_queue)
        except Exception as e:
            log.error(f"Erreur vision : {e}")

    async def _ha_control(self, entity_id: str, service: str):
        """Contrôle Home Assistant."""
        ha_url = os.getenv("HA_URL")
        ha_token = os.getenv("HA_TOKEN")
        if not ha_url or not ha_token:
            log.warning("HA_URL ou HA_TOKEN manquant dans le .env")
            return

        domain = entity_id.split('.')[0]
        url = f"{ha_url}/api/services/{domain}/{service}"
        headers = {
            "Authorization": f"Bearer {ha_token}",
            "Content-Type": "application/json",
        }
        payload = {"entity_id": entity_id}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        log.info(f"🏠 HA : {service} sur {entity_id} réussi.")
                    else:
                        log.error(f"🏠 HA Error : {resp.status}")
        except Exception as e:
            log.error(f"HA Connection Error : {e}")

    async def _process_and_respond(self, user_text: str):
        """
        Orchestrateur Agentique : Perception -> Classification -> Raisonnement -> Action.
        """
        # 1. Classification d'Intention
        intent = IntentClassifier.classify(user_text)
        log.info(f"🎯 Intention détectée : {intent}")

        sentence_endings = re.compile(r'(?<=[.!?])\s+')
        current_sentence = ""
        full_response = ""
        log.info("🤖 Jarvis réfléchit...")

        if self.signals:
            self.signals.thinking_state_changed.emit(True)

        # 2. Construction du prompt enrichi avec l'état
        prompt_with_state = user_text
        if intent == JarvisState.CHAT:
            prompt_with_state += " (Note: Réponds simplement à la discussion, aucune action système n'est requise ici.)"

        self.history.append({"role": "user", "content": prompt_with_state})

        try:
            async for token in self.llm.generate_stream(self.history):
                current_sentence += token
                full_response += token

                # Détection de commandes déterministe
                if "]" in token:
                    extracted_cmds = CommandParser.extract_all(full_response)
                    for cmd_tag in extracted_cmds:
                        if cmd_tag not in full_response: continue # Déjà traitée

                        cmd_result = self._execute_command(cmd_tag)

                        # Injection du résultat dans l'historique pour le feedback loop du LLM
                        self.history.append({"role": "system", "content": f"Command Result: {cmd_result}"})

                        # Retrait du tag pour le TTS
                        current_sentence = current_sentence.replace(cmd_tag, "")
                        full_response = full_response.replace(cmd_tag, "")

                if any(c in token for c in ".!?"):
                    parts = sentence_endings.split(current_sentence)
                    if len(parts) > 1:
                        for i in range(len(parts) - 1):
                            sentence_to_speak = parts[i].strip()
                            if sentence_to_speak:
                                # Retirer les éventuels restes de tags
                                sentence_to_speak = re.sub(r"\[CMD:.*?\]", "", sentence_to_speak).strip()
                                if sentence_to_speak:
                                    log.info(f"🎙️ TTS (phrase) : {sentence_to_speak}")
                                    await self.tts.speak(sentence_to_speak, self.audio_queue)
                        current_sentence = parts[-1]

            if current_sentence.strip():
                log.info(f"🎙️ TTS (final) : {current_sentence.strip()}")
                await self.tts.speak(current_sentence.strip(), self.audio_queue)

            if self.signals:
                self.signals.thinking_state_changed.emit(False)

            # Ajouter la réponse complète à l'historique
            if full_response.strip():
                self.history.append({"role": "assistant", "content": full_response.strip()})
                # Limiter l'historique pour éviter les prompts trop longs (on garde le system prompt + X derniers messages)
                if len(self.history) > self.history_limit:
                    self.history = [self.history[0]] + self.history[-(self.history_limit-1):]

        except asyncio.CancelledError:
            log.debug("Tâche de réponse annulée.")
        except Exception as e:
            log.error(f"Erreur dans le cycle de réponse : {e}")
            if self.signals:
                self.signals.thinking_state_changed.emit(False)


# ----------------------------------------------------------------------
# Entrée du script avec intégration PyQt6 + Asyncio
# ----------------------------------------------------------------------
class JarvisWorker(QThread):
    def __init__(self, signals):
        super().__init__()
        self.signals = signals

    def run(self):
        asyncio.run(self.run_async())

    async def run_async(self):
        try:
            self.jarvis = Jarvis(signals=self.signals)
            await self.jarvis.run()
        except Exception as e:
            log.error(f"Erreur dans le worker Jarvis : {e}")


def main():
    app = QApplication(sys.argv)

    # HUD
    reactor = ArcReactor()

    # Centrer le HUD sur l'écran principal
    screen = app.primaryScreen().availableGeometry()
    reactor.move(screen.center().x() - reactor.width() // 2,
                 screen.center().y() - reactor.height() // 2)

    reactor.show()

    # Signaux
    signals = JarvisSignals()
    signals.transcription_received.connect(reactor.set_text)
    signals.thinking_state_changed.connect(reactor.set_thinking)

    # Démarrage de Jarvis dans un thread séparé
    worker = JarvisWorker(signals)
    worker.start()

    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        log.info("\n👋 Arrêt demandé par l'utilisateur. Au revoir !")


if __name__ == "__main__":
    main()
