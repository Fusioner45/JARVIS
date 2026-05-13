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
from collections import deque
from typing import AsyncGenerator, List, Dict

import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QGraphicsDropShadowEffect
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QPropertyAnimation, QEasingCurve, QPointF, QRectF
from PyQt6.QtGui import QColor, QPalette, QFont, QPainter, QRadialGradient, QPen, QBrush
import webbrowser
import os
import pyautogui
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
import aiohttp
import edge_tts
from pydub import AudioSegment  # décodage MP3 → PCM

# ----------------------------------------------------------------------
# Persistence: SQLite Memory Core
# ----------------------------------------------------------------------
class JarvisMemory:
    def __init__(self, db_path="memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact_type TEXT,
                    content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def save_memory(self, fact_type: str, content: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO memory (fact_type, content) VALUES (?, ?)", (fact_type, content))
            conn.commit()
            log.info(f"🧠 Mémoire sauvegardée : [{fact_type}] {content}")

    def query_memory(self, search_term: str) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT content FROM memory WHERE content LIKE ? OR fact_type LIKE ?",
                                 (f"%{search_term}%", f"%{search_term}%"))
            return [row[0] for row in cursor.fetchall()]

    def delete_memory(self, search_term: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM memory WHERE content LIKE ? OR fact_type LIKE ?",
                         (f"%{search_term}%", f"%{search_term}%"))
            conn.commit()
            log.info(f"🗑️ Mémoire supprimée pour : {search_term}")

    def get_all_context(self) -> str:
        """Récupère un résumé de tous les faits pour le prompt système."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT fact_type, content FROM memory ORDER BY timestamp DESC LIMIT 20")
            facts = [f"- {ft}: {c}" for ft, c in cursor.fetchall()]
            return "\n".join(facts) if facts else "Aucun fait mémorisé pour le moment."

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

    def is_speech(self, frame: bytes, threshold: float = 0.5) -> bool:
        """
        Retourne True si la frame contient de la voix (probabilité > threshold).
        """
        audio_int16 = np.frombuffer(frame, dtype=np.int16)
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

    async def transcribe(self, audio_frames: List[bytes]) -> str:
        """
        Transcrit une liste de frames PCM16 en texte.
        Fonctionne en thread séparé pour ne pas bloquer la boucle asyncio.
        """
        loop = asyncio.get_running_loop()

        def _sync_transcribe():
            # Concaténer toutes les frames en un seul np.ndarray float32 [-1,1]
            audio_np = np.frombuffer(b"".join(audio_frames), dtype=np.int16).astype(np.float32) / 32768.0
            # faster‑whisper attend un tableau 1‑D
            segments, info = self.model.transcribe(
                audio_np,
                language="fr",
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

    async def speak(self, text: str, audio_queue: asyncio.Queue):
        """
        Lit le texte à voix haute grâce à edge‑tts.
        Accumule tout l'audio d'une phrase avant de le décoder pour éviter les erreurs FFMPEG.
        """
        if not text:
            return

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
            await audio_queue.put(pcm_array)


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
        try:
            # Rotation ring
            self.angle_outer = (self.angle_outer + (4 if self.is_thinking else 1)) % 360
            self.pulse_inner += 0.15 if self.is_thinking else 0.05

            # Typewriter effect
            if len(self.current_text) < len(self.target_text):
                self.current_text += self.target_text[len(self.current_text)]
                self.label.setText(self.current_text)

            self.update()
        except Exception:
            self.timer.stop()

    def paintEvent(self, event):
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
# Orchestrateur principal – boucle async
# ----------------------------------------------------------------------
class Jarvis:
    def __init__(self, signals: JarvisSignals = None):
        self.signals = signals
        self.memory = JarvisMemory()
        self.vad = VoiceActivityDetector()
        self.stt = SpeechToText()
        self.llm = LlmClient()
        self.tts = TextToSpeech()

        # Buffers
        self._pre_roll_buffer: deque[bytes] = deque(maxlen=10) # ~320ms de "passé"
        self._speech_buffer: List[bytes] = []
        self._silence_count = 0
        self._is_speaking = False

        # Audio output synchronization
        self._audio_thread_lock = threading.Lock()
        self._current_audio_chunk = np.array([], dtype=np.int16)

        # Initial context loading from memory
        context = self.memory.get_all_context()
        self.history = [
            {"role": "system", "content": f"""Tu es JARVIS, un assistant personnel français intelligent et proactif.
Tu dois TOUJOURS répondre en français. Tes réponses doivent être concises et adaptées à une interaction vocale.

CONTEXTE MÉMOIRE (Faits dont tu dois te souvenir) :
{context}

ACTIONS DISPONIBLES (Inclus-les dans ta réponse si nécessaire) :
- [CMD: OPEN_APP('nom')] : Pour ouvrir une application.
- [CMD: SEARCH_WEB('requête')] : Pour faire une recherche.
- [CMD: SAVE_FACT('type', 'contenu')] : Pour mémoriser une information importante.
- [CMD: DELETE_FACT('recherche')] : Pour supprimer un fait de la mémoire.
- [CMD: MIDI('commande')] : Placeholder pour le contrôle musical.

Exemple : "Très bien monsieur, je lance Spotify. [CMD: OPEN_APP('spotify')]"
"""}
        ]
        self.history_limit = 10

    async def _audio_listener(self) -> AsyncGenerator[str, None]:
        """
        Écoute en continu le microphone, applique le VAD,
        et yield une transcription dès qu'une fin d'énoncé est détectée.
        """
        async for frame in audio_frame_generator():
            is_speech = self.vad.is_speech(frame)

            if is_speech:
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
                            transcript = await self.stt.transcribe(self._speech_buffer)
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
        Boucle principale avec streaming LLM et TTS phrase par phrase.
        """
        log.info("🚀 Jarvis V2 démarré – dites quelque chose !")

        # File d'attente pour l'audio PCM à jouer
        self.audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

        # Stream sounddevice avec callback pour une lecture fluide sans stuttering
        def audio_callback(outdata, frames, time, status):
            if status:
                log.warning(f"Audio output status: {status}")

            # On essaie de récupérer de la donnée du buffer interne
            data = self._get_next_audio_chunk(frames)
            if data is not None:
                outdata[:len(data), 0] = data
                if len(data) < frames:
                    outdata[len(data):, 0] = 0
            else:
                outdata.fill(0)

        # Buffer interne pour le callback
        self._current_audio_chunk = np.array([], dtype=np.int16)

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

        worker_task = asyncio.create_task(audio_worker())

        try:
            async for user_text in self._audio_listener():
                # Barge-in : annuler la réponse et vider l'audio
                if hasattr(self, "_response_task") and not self._response_task.done():
                    log.info("🚫 Interruption (Barge-in) – on coupe la parole.")
                    self._response_task.cancel()

                # Vider le buffer audio pour arrêter de parler immédiatement
                self._clear_audio_buffer()
                while not self.audio_queue.empty():
                    try: self.audio_queue.get_nowait(); self.audio_queue.task_done()
                    except asyncio.QueueEmpty: break

                self._response_task = asyncio.create_task(self._process_and_respond(user_text))
        finally:
            output_stream.stop()
            output_stream.close()
            await self.audio_queue.put(None)
            await worker_task
            await self.llm.close()
            log.info("🔌 Session Ollama fermée.")

    def _append_audio_chunk(self, chunk):
        with self._audio_thread_lock:
            if self._current_audio_chunk.size == 0:
                self._current_audio_chunk = chunk
            else:
                self._current_audio_chunk = np.concatenate([self._current_audio_chunk, chunk])

    def _get_next_audio_chunk(self, frames):
        with self._audio_thread_lock:
            if self._current_audio_chunk.size == 0:
                return None

            take = min(frames, self._current_audio_chunk.size)
            chunk = self._current_audio_chunk[:take]
            self._current_audio_chunk = self._current_audio_chunk[take:]
            return chunk

    def _clear_audio_buffer(self):
        with self._audio_thread_lock:
            self._current_audio_chunk = np.array([], dtype=np.int16)

    def _execute_command(self, cmd_tag: str):
        """Analyse et exécute un tag [CMD: ...]."""
        try:
            # Extraction du nom de la commande et de ses arguments
            content = cmd_tag.replace("[CMD:", "").replace("]", "").strip()
            # On cherche qqc comme OPEN_APP('spotify')
            match = re.match(r"(\w+)\((.*)\)", content)
            if not match: return

            cmd_name = match.group(1)
            # Nettoyage rudimentaire des quotes
            args = [a.strip().strip("'").strip('"') for a in match.group(2).split(",")]

            log.info(f"🚀 Exécution commande : {cmd_name} avec args {args}")

            if cmd_name == "OPEN_APP":
                # Sur Windows, on peut souvent juste lancer le nom de l'exe
                os.system(f"start {args[0]}")
                if "spotify" in args[0].lower():
                    # Petite automatisation pour lancer la lecture
                    def _spotify_play():
                        time.sleep(3)
                        pyautogui.press('enter')
                    threading.Thread(target=_spotify_play, daemon=True).start()
            elif cmd_name == "SEARCH_WEB":
                webbrowser.open(f"https://www.google.com/search?q={args[0]}")
            elif cmd_name == "SAVE_FACT" and len(args) >= 2:
                self.memory.save_memory(args[0], args[1])
            elif cmd_name == "DELETE_FACT":
                self.memory.delete_memory(args[0])
            elif cmd_name == "MIDI":
                log.info(f"🎹 MIDI Placeholder: {args[0]}")
        except Exception as e:
            log.error(f"Erreur exécution commande {cmd_tag}: {e}")

    async def _process_and_respond(self, user_text: str):
        """
        Gère le flux : LLM stream -> Découpage en phrases -> TTS.
        """
        sentence_endings = re.compile(r'(?<=[.!?])\s+')
        current_sentence = ""
        full_response = ""
        log.info("🤖 Jarvis réfléchit...")

        if self.signals:
            self.signals.thinking_state_changed.emit(True)

        # Ajouter le message utilisateur à l'historique
        self.history.append({"role": "user", "content": user_text})

        try:
            async for token in self.llm.generate_stream(self.history):
                current_sentence += token
                full_response += token

                # Détection de commandes au fil de l'eau
                if "]" in token and "[CMD:" in full_response:
                    cmd_match = re.search(r"(\[CMD:.*?\])", full_response)
                    if cmd_match:
                        cmd_tag = cmd_match.group(1)
                        self._execute_command(cmd_tag)
                        # On retire le tag du texte pour ne pas que le TTS le lise
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
