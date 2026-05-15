import os
import time
import asyncio
import logging
import re
import queue
import aiohttp
import io
import pyautogui
import numpy as np
import sounddevice as sd
from collections import deque
from typing import AsyncGenerator

from jarvis.utils.config import (
    SAMPLE_RATE, FRAME_SIZE, SILENCE_FRAMES_THRESHOLD, OLLAMA_HOST
)
from jarvis.utils.logger import system_log as log
from jarvis.utils.helpers import AudioController, GpuMonitor
from jarvis.core.context import JarvisContext
from jarvis.core.states import JarvisState
from jarvis.perception.audio import audio_frame_generator, VoiceActivityDetector
from jarvis.perception.stt import SpeechToText
from jarvis.intelligence.memory import JarvisMemory
from jarvis.intelligence.llm import LlmClient, IntentClassifier
from jarvis.actions.parser import CommandParser
from jarvis.actions.executor import CommandExecutor, ActionGuard
from jarvis.tts.engine import TextToSpeech

class Jarvis:
    def __init__(self, signals=None):
        self.signals = signals
        self.context = JarvisContext()
        self.memory = JarvisMemory()
        self.audio_ctrl = AudioController()
        self.gpu_mon = GpuMonitor()
        self.guard = ActionGuard()
        self.vad = VoiceActivityDetector()
        self.stt = SpeechToText()
        self.llm = LlmClient()
        self.tts = TextToSpeech()

        self._pre_roll_buffer = deque(maxlen=10)
        self._speech_buffer = []
        self._silence_count = 0
        self._is_listening = False
        self.playback_queue = queue.Queue(maxsize=100)

        self.context.user_name = self.memory.get_user_name()

        self.history = [
            {"role": "system", "content": f"""Tu es JARVIS. Réponds en Français, ultra-concise.
Ton utilisateur est {self.context.user_name}.

CONTEXTE MÉMOIRE : {self.memory.get_all_context()}

ACTIONS DISPONIBLES :
- [CMD: OPEN_APP('nom')]
- [CMD: SEARCH_WEB('requête')]
- [CMD: PLAY_MUSIC('mot-clé')]
- [CMD: SAVE_FACT('type', 'contenu')]
- [CMD: DELETE_FACT('terme')]
- [CMD: SAVE_TASK('tâche', 'échéance')]
- [CMD: GET_GPU_TEMP()]
- [CMD: SPLIT_SCREEN('app1', 'app2')]
- [CMD: WORK_MODE()]
- [CMD: INDEX_PDF('chemin')]
- [CMD: SCREENSHOT_ANALYZE()]
- [CMD: HA_CONTROL('entité', 'service')]
"""}
        ]

    async def _audio_listener(self) -> AsyncGenerator[str, None]:
        async for frame in audio_frame_generator(self.context):
            is_speech = self.vad.is_speech(frame)
            if is_speech:
                if not self._is_listening:
                    self._is_listening = True
                    self.context.set_state(JarvisState.LISTENING)
                    if hasattr(self, "_response_task") and not self._response_task.done():
                        self._response_task.cancel()
                        self._clear_playback()
                    self._speech_buffer = list(self._pre_roll_buffer)
                self._speech_buffer.append(frame)
                self._silence_count = 0
            else:
                if self._is_listening:
                    self._speech_buffer.append(frame)
                    self._silence_count += 1
                    if self._silence_count > SILENCE_FRAMES_THRESHOLD:
                        self._is_listening = False
                        self.context.set_state(JarvisState.IDLE)
                        if self._speech_buffer:
                            self.audio_ctrl.set_ducking(True)
                            transcript = await self.stt.transcribe(self._speech_buffer)
                            self.audio_ctrl.set_ducking(False)
                            if transcript:
                                if self.signals: self.signals.transcription_received.emit(transcript)
                                yield transcript
                        self._speech_buffer = []
                        self._silence_count = 0
                else:
                    self._pre_roll_buffer.append(frame)

    async def run(self):
        def audio_callback(outdata, frames, time, status):
            try:
                data = self.playback_queue.get_nowait()
                outdata[:, 0] = data
            except queue.Empty: outdata.fill(0)

        output_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            callback=audio_callback, blocksize=FRAME_SIZE,
        )
        output_stream.start()

        async def playback_manager():
            while True:
                chunk = await self.context.audio_output_queue.get()
                if chunk is None: break
                for i in range(0, len(chunk), FRAME_SIZE):
                    seg = chunk[i:i+FRAME_SIZE]
                    if len(seg) < FRAME_SIZE:
                        seg = np.concatenate([seg, np.zeros(FRAME_SIZE - len(seg), dtype=np.int16)])
                    self.playback_queue.put(seg)
                if self.context.audio_output_queue.empty():
                    self.context.is_speaking = False

        playback_task = asyncio.create_task(playback_manager())
        await self.tts.speak(f"Bonjour {self.context.user_name}.", self.context)

        try:
            async for user_text in self._audio_listener():
                self._response_task = asyncio.create_task(self._process_and_respond(user_text))
                await self._response_task
        finally:
            output_stream.stop()
            output_stream.close()
            await self.context.audio_output_queue.put(None)
            await playback_task
            await self.llm.close()

    def _clear_playback(self):
        while not self.playback_queue.empty():
            try: self.playback_queue.get_nowait()
            except: break
        self.context.is_speaking = False

    async def _execute_complex_command(self, name, args):
        cmd_id = f"{name}:{args}"
        if self.guard.is_blocked(cmd_id): return "BLOCKED"

        res = CommandExecutor.execute(name, args, self)

        if res == "HANDLED_BY_ORCHESTRATOR":
            if name == "SAVE_FACT": self.memory.save_memory(args[0], args[1]); res = "SUCCESS"
            elif name == "DELETE_FACT": self.memory.delete_memory(args[0]); res = "SUCCESS"
            elif name == "SAVE_TASK": self.memory.save_task(args[0], args[1] if len(args) > 1 else None); res = "SUCCESS"
            elif name == "INDEX_PDF": self.memory.index_pdf(args[0]); res = "SUCCESS"
            elif name == "SET_VOICE_MORPH": self.tts.use_morphing = (args[0].lower() == "true"); res = "SUCCESS"
            elif name == "WORK_MODE":
                CommandExecutor.execute("OPEN_APP", ["vscode"])
                self.playback_queue.put(np.zeros(10, dtype=np.int16)) # dummy
                res = "SUCCESS"
            elif name == "HA_CONTROL": await self._ha_control(args[0], args[1]); res = "SUCCESS"
            elif name == "SCREENSHOT_ANALYZE": await self._screenshot_and_analyze(); res = "SUCCESS"

        self.guard.record(cmd_id, "SUCCESS" in res)
        return res

    async def _ha_control(self, entity_id, service):
        ha_url = os.getenv("HA_URL")
        ha_token = os.getenv("HA_TOKEN")
        if not ha_url or not ha_token: return
        url = f"{ha_url}/api/services/{entity_id.split('.')[0]}/{service}"
        headers = {"Authorization": f"Bearer {ha_token}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            await session.post(url, json={"entity_id": entity_id}, headers=headers)

    async def _screenshot_and_analyze(self):
        import base64
        screenshot = pyautogui.screenshot()
        img_byte_arr = io.BytesIO()
        screenshot.save(img_byte_arr, format='PNG')
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        payload = {"model": "moondream", "prompt": "Décris cet écran.", "images": [img_base64], "stream": False}
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{OLLAMA_HOST}/api/generate", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    await self.tts.speak(data.get("response", ""), self.context)

    async def _process_and_respond(self, user_text: str):
        self.context.set_state(JarvisState.THINKING)
        if self.signals: self.signals.thinking_state_changed.emit(True)
        self.history.append({"role": "user", "content": user_text})
        full_response = ""
        current_sentence = ""
        sentence_endings = re.compile(r'(?<=[.!?])\s+')

        try:
            async for token in self.llm.generate_stream(self.history):
                current_sentence += token
                full_response += token
                if "]" in token:
                    cmds = CommandParser.extract_all(full_response)
                    for cmd_tag in cmds:
                        name, args = CommandParser.parse_call(cmd_tag)
                        res = await self._execute_complex_command(name, args)
                        self.history.append({"role": "system", "content": f"Result: {res}"})
                        current_sentence = current_sentence.replace(cmd_tag, "")
                        full_response = full_response.replace(cmd_tag, "")
                if any(c in token for c in ".!?"):
                    parts = sentence_endings.split(current_sentence)
                    if len(parts) > 1:
                        for i in range(len(parts)-1):
                            s = re.sub(r"\[CMD:.*?\]", "", parts[i]).strip()
                            if s: await self.tts.speak(s, self.context)
                        current_sentence = parts[-1]
            if current_sentence.strip():
                s = re.sub(r"\[CMD:.*?\]", "", current_sentence).strip()
                if s: await self.tts.speak(s, self.context)
            self.history.append({"role": "assistant", "content": full_response})
            self.context.set_state(JarvisState.IDLE)
            if self.signals: self.signals.thinking_state_changed.emit(False)
        except asyncio.CancelledError: pass
        except Exception as e: log.error(f"Error: {e}")
