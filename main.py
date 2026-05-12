#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Jarvis V2 – Assistant vocal ultra‑rapide (Windows 11, RTX 3070 Ti)

Architecture :
    ├─ STT  : faster‑whisper (small) + webrtcvad (VAD)  → CUDA
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
from collections import deque
from typing import AsyncGenerator, List

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
import aiohttp
import edge_tts
from pydub import AudioSegment  # décodage MP3 → PCM

# ----------------------------------------------------------------------
# Configuration (à adapter si besoin)
# ----------------------------------------------------------------------
SAMPLE_RATE = 16000               # Hz – required by webrtcvad & faster‑whisper
FRAME_MS = 30                     # ms – size of each audio frame for VAD
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)  # samples per frame
VAD_MODE = 2                      # 0‑3, 2 = bonne compromis sensibilité/robustesse
SILENCE_FRAMES_THRESHOLD = 30     # nombre de frames silencieuses pour finir une utterance
WHISPER_MODEL_SIZE = "small"      # tiny, base, small, medium, large‑v2 …
WHISPER_DEVICE = "cuda"           # on utilise le GPU
WHISPER_COMPUTE_TYPE = "float16"  # optimum pour RTX 30xx
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
# VAD wrapper – webrtcvad works on 10,20,30 ms frames
# ----------------------------------------------------------------------
class VoiceActivityDetector:
    def __init__(self, mode: int = VAD_MODE, sample_rate: int = SAMPLE_RATE):
        self.vad = webrtcvad.Vad(mode)
        self.sample_rate = sample_rate

    def is_speech(self, frame: bytes) -> bool:
        """
        Retourne True si la frame contient de la voix.
        La frame doit être exactement FRAME_SIZE * 2 bytes (int16).
        """
        try:
            return self.vad.is_speech(frame, self.sample_rate)
        except Exception as e:
            log.debug(f"VAD error: {e}")
            return False


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

    async def generate(self, prompt: str) -> str:
        """
        Envoie le prompt à Ollama et retourne la réponse complète.
        Utilise l'endpoint /v1/chat/completions (format OpenAI).
        """
        await self._ensure_session()
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Tu es Jarvis, un assistant vocal utile et concis."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 512,
        }
        try:
            async with self.session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log.error(f"Erreur lors de l'appel à Ollama : {e}")
            return "Désolé, je n'ai pas pu obtenir de réponse."


# ----------------------------------------------------------------------
# TTS – edge‑tts (voix FR) → MP3 → PCM via pydub/ffmpeg
# ----------------------------------------------------------------------
class TextToSpeech:
    def __init__(self, voice: str = EDGE_TTS_VOICE):
        self.voice = voice

    async def speak(self, text: str):
        """
        Lit le texte à voix haute grâce à edge‑tts.
        Le flux MP3 reçu est décodé en PCM 16‑bit mono à SAMPLE_RATE
        puis envoyé directement à sounddevice via un OutputStream.
        """
        if not text:
            return

        communicate = edge_tts.Communicate(text, voice=self.voice)

        # Ouvrir un flux de sortie en mode blocking (écriture directe)
        with sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",  # on écrit des int16 directement
        ) as stream:
            async for chunk in communicate.stream():
                if chunk["type"] != "audio":
                    continue  # on ignore les événements WordBoundary, etc.
                mp3_bytes: bytes = chunk["data"]  # <-- le champ contenant le MP3

                try:
                    # Décodage MP3 → PCM (int16, mono, SAMPLE_RATE)
                    audio_segment = AudioSegment.from_file(
                        io.BytesIO(mp3_bytes), format="mp3"
                    )
                    audio_segment = audio_segment.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
                    pcm = audio_segment.raw_data  # bytes, little‑endian int16
                except Exception as e:
                    log.error(f"Erreur de décodage MP3 : {e}")
                    # On continue avec du silence pour ne pas bloquer la lecture
                    pcm = b"\x00" * (FRAME_SIZE * 2)  # un frame de silence

                # écriture dans le stream (sounddevice accepte les bytes ou np.ndarray)
                stream.write(pcm)  # type: ignore[arg-type]  # sounddevice accepte bytes


# ----------------------------------------------------------------------
# Orchestrateur principal – boucle async
# ----------------------------------------------------------------------
class Jarvis:
    def __init__(self):
        self.vad = VoiceActivityDetector()
        self.stt = SpeechToText()
        self.llm = LlmClient()
        self.tts = TextToSpeech()

        # Buffer pour accumuler les frames de parole détectée
        self._speech_buffer: deque[bytes] = deque()
        self._silence_count = 0

    async def _audio_listener(self) -> AsyncGenerator[str, None]:
        """
        Écoute en continu le microphone, applique le VAD,
        et yield une transcription dès qu'une fin d'énoncé est détectée.
        """
        async for frame in audio_frame_generator():
            is_speech = self.vad.is_speech(frame)

            if is_speech:
                self._speech_buffer.append(frame)
                self._silence_count = 0
                log.debug("🔊 Voix détectée – buffer en cours")
            else:
                # Pas de voix dans cette frame
                self._speech_buffer.append(frame)  # on garde quand même un petit dépassement
                self._silence_count += 1
                if self._silence_count > SILENCE_FRAMES_THRESHOLD:
                    # Fin d'énoncé : on transmet le buffer puis on le vide
                    if self._speech_buffer:
                        log.info("⏹️ Silence détecté – transcription en cours…")
                        transcript = await self.stt.transcribe(list(self._speech_buffer))
                        if transcript:
                            log.info(f"🗣️ Transcription : {transcript}")
                            yield transcript
                        else:
                            log.info("🗣️ Transcription vide (probablement bruit).")
                    # Reset pour la prochaine phrase
                    self._speech_buffer.clear()
                    self._silence_count = 0

    async def run(self):
        """
        Boucle principale :
            1. Écoute (STT)
            2. Envoie le texte au LLM
            3. Fais parler la réponse (TTS)
        Tout se fait en parallèle grâce à des tâches asyncio.
        """
        log.info("🚀 Jarvis V2 démarré – dites quelque chose !")
        try:
            async for user_text in self._audio_listener():
                # Annuler les tâches éventuellement en cours (nouvelle utterance)
                for task_name in ("think", "speak"):
                    task = getattr(self, f"_{task_name}_task", None)
                    if task and not task.done():
                        task.cancel()

                # Étape 2 : réflexion (LLM)
                self._think_task = asyncio.create_task(self.llm.generate(user_text))
                try:
                    ai_response = await self._think_task
                except asyncio.CancelledError:
                    continue  # nouvelle utterance arrivée, on recommence
                except Exception as e:
                    log.error(f"Erreur LLM : {e}")
                    ai_response = "Désolé, une erreur est survenue."

                log.info(f"🤖 Réponse IA : {ai_response}")

                # Étape 3 : parole (TTS)
                self._speak_task = asyncio.create_task(self.tts.speak(ai_response))
                await self._speak_task  # on attend la fin de la lecture avant de reprendre l'écoute
                # L'écoute tourne en tâche de fond via le générateur audio_frame_generator,
                # donc aucune tâche séparée n'est nécessaire ici.
        finally:
            # S'assurer que le session HTTP est bien fermée même en cas d'erreur
            await self.llm.close()
            log.info("🔌 Session Ollama fermée.")


# ----------------------------------------------------------------------
# Entrée du script
# ----------------------------------------------------------------------
def main():
    try:
        jarvis = Jarvis()
        asyncio.run(jarvis.run())
    except KeyboardInterrupt:
        log.info("\n👋 Arrêt demandé par l'utilisateur. Au revoir !")
    except Exception as exc:
        log.exception(f"Erreur fatale : {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
