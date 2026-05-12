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
from collections import deque
from typing import AsyncGenerator, List

import numpy as np
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
WHISPER_MODEL_SIZE = "small"      # tiny, base, small, medium, large‑v2 …
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

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Envoie le prompt à Ollama et yield les morceaux de texte au fur et à mesure.
        Utilise l'endpoint /v1/chat/completions avec stream=True.
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
            "stream": True,
        }
        try:
            async with self.session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=60)
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

    async def speak(self, text: str, stream: sd.OutputStream):
        """
        Lit le texte à voix haute grâce à edge‑tts.
        Écrit directement dans le stream sounddevice fourni.
        """
        if not text:
            return

        communicate = edge_tts.Communicate(text, voice=self.voice)
        loop = asyncio.get_running_loop()

        async for chunk in communicate.stream():
            if chunk["type"] != "audio":
                continue
            mp3_bytes: bytes = chunk["data"]

            try:
                # Décodage MP3 → PCM (int16, mono, SAMPLE_RATE)
                audio_segment = AudioSegment.from_file(
                    io.BytesIO(mp3_bytes), format="mp3"
                )
                audio_segment = audio_segment.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
                pcm = audio_segment.raw_data
            except Exception as e:
                log.error(f"Erreur de décodage MP3 : {e}")
                continue

            # Écrire dans le stream sounddevice via un exécuteur pour ne pas bloquer l'event loop
            await loop.run_in_executor(None, stream.write, pcm)


# ----------------------------------------------------------------------
# Orchestrateur principal – boucle async
# ----------------------------------------------------------------------
class Jarvis:
    def __init__(self):
        self.vad = VoiceActivityDetector()
        self.stt = SpeechToText()
        self.llm = LlmClient()
        self.tts = TextToSpeech()

        # Buffers
        self._pre_roll_buffer: deque[bytes] = deque(maxlen=10) # ~320ms de "passé"
        self._speech_buffer: List[bytes] = []
        self._silence_count = 0
        self._is_speaking = False

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

        # Stream de sortie partagé pour le TTS
        output_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )
        output_stream.start()

        try:
            async for user_text in self._audio_listener():
                # Barge-in : annuler la réponse en cours
                if hasattr(self, "_response_task") and not self._response_task.done():
                    log.info("🚫 Interruption (Barge-in) – on coupe la parole.")
                    self._response_task.cancel()
                    # On ne peut pas facilement vider le buffer matériel de sounddevice ici,
                    # mais arrêter d'écrire dedans est déjà un bon début.

                self._response_task = asyncio.create_task(self._process_and_respond(user_text, output_stream))
        finally:
            output_stream.stop()
            output_stream.close()
            await self.llm.close()
            log.info("🔌 Session Ollama fermée.")

    async def _process_and_respond(self, user_text: str, output_stream: sd.OutputStream):
        """
        Gère le flux : LLM stream -> Découpage en phrases -> TTS.
        """
        # On découpe sur la ponctuation suivie d'un espace ou fin de ligne
        sentence_endings = re.compile(r'(?<=[.!?])\s+')

        current_sentence = ""
        log.info("🤖 Jarvis réfléchit...")

        try:
            async for token in self.llm.generate_stream(user_text):
                current_sentence += token

                # Si on a un signe de ponctuation fort, on tente de découper
                if any(c in token for c in ".!?"):
                    parts = sentence_endings.split(current_sentence)
                    # Si on a au moins une phrase complète (parts[0]) et un reliquat (parts[1...])
                    if len(parts) > 1:
                        for i in range(len(parts) - 1):
                            sentence_to_speak = parts[i].strip()
                            if sentence_to_speak:
                                log.info(f"🎙️ TTS (phrase) : {sentence_to_speak}")
                                await self.tts.speak(sentence_to_speak, output_stream)
                        current_sentence = parts[-1]

            # Parler le reste (dernière phrase sans ponctuation finale peut-être)
            if current_sentence.strip():
                log.info(f"🎙️ TTS (final) : {current_sentence.strip()}")
                await self.tts.speak(current_sentence.strip(), output_stream)

        except asyncio.CancelledError:
            log.debug("Tâche de réponse annulée.")
            # On ne ferme pas la session ici car elle est partagée
        except Exception as e:
            log.error(f"Erreur dans le cycle de réponse : {e}")


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
