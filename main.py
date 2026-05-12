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
import threading
import time
from collections import deque
from typing import AsyncGenerator, List

import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QColor, QPalette, QFont, QPainter, QRadialGradient
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
import aiohttp
import edge_tts
from pydub import AudioSegment  # décodage MP3 → PCM

# ----------------------------------------------------------------------
# Configuration (à adapter si besoin)
# ----------------------------------------------------------------------
SAMPLE_RATE = 16000               # Hz – required by webrtcvad & faster‑whisper
FRAME_MS = 32                     # ms – size of each audio frame for VAD
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)  # samples per frame
# VAD_MODE deleted                      # 0‑3, 2 = bonne compromis sensibilité/robustesse
SILENCE_FRAMES_THRESHOLD = 15     # nombre de frames silencieuses pour finir une utterance
WHISPER_MODEL_SIZE = "distil-large-v3"      # tiny, base, small, medium, large‑v2 …
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
                initial_prompt="Ceci est une conversation en français entre un humain et un assistant nommé Jarvis.",
                beam_size=5,
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
        """
        await self._ensure_session()
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 512,
            "stream": True,
        }
        try:
            # On utilise un timeout plus long pour la connexion et on laisse le stream s'écouler
            async with self.session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(connect=5, total=120)
            ) as resp:
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
        except Exception as e:
            log.error(f"Erreur lors du stream Ollama : {e}")
            yield "Désolé, je n'ai pas pu obtenir de réponse."


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
# UI PyQt6 - Cyberpunk HUD
# ----------------------------------------------------------------------
class ArcReactor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(400, 400)

        self.glow_radius = 50
        self.glow_color = QColor(0, 255, 255, 150) # Cyan cyberpunk

        # Layout pour la transcription
        layout = QVBoxLayout(self)
        layout.addStretch()
        self.label = QLabel("", self)
        self.label.setStyleSheet("color: #00ffff; font-family: 'Consolas'; font-size: 16px; background-color: rgba(0,0,0,100); padding: 5px;")
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

        # Animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(50)
        self.pulse_val = 0
        self.is_thinking = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        center = self.rect().center()

        # Calcul du pulse
        if self.is_thinking:
            self.pulse_val += 0.2
        else:
            self.pulse_val += 0.05

        dynamic_radius = self.glow_radius + (15 * np.sin(self.pulse_val))

        # Gradient pour l'effet Arc Reactor
        gradient = QRadialGradient(center, dynamic_radius)
        if self.is_thinking:
            gradient.setColorAt(0, QColor(255, 0, 255, 200)) # Magenta quand il réfléchit
        else:
            gradient.setColorAt(0, QColor(0, 255, 255, 200))

        gradient.setColorAt(0.5, QColor(0, 100, 255, 100))
        gradient.setColorAt(1, QColor(0, 0, 0, 0))

        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, int(dynamic_radius), int(dynamic_radius))

    def set_text(self, text):
        self.label.setText(text)

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

        # Conversation context
        self.history = [
            {"role": "system", "content": "Tu es JARVIS, un assistant personnel français. Tu dois TOUJOURS répondre en français, peu importe la langue utilisée par l'utilisateur. Tes réponses doivent être concises et adaptées à une interaction vocale."}
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
                if any(c in token for c in ".!?"):
                    parts = sentence_endings.split(current_sentence)
                    if len(parts) > 1:
                        for i in range(len(parts) - 1):
                            sentence_to_speak = parts[i].strip()
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
