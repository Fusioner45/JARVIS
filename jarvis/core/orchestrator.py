import asyncio
import time
import re
import queue
import numpy as np
import sounddevice as sd
from collections import deque
from jarvis.utils.config import SAMPLE_RATE, FRAME_SIZE, SILENCE_FRAMES_THRESHOLD
from jarvis.utils.logger import system_log as log
from jarvis.utils.helpers import AudioController, GpuMonitor
from jarvis.core.context import JarvisContext
from jarvis.core.states import JarvisState
from jarvis.perception.audio import audio_frame_generator, VoiceActivityDetector
from jarvis.perception.stt import SpeechToText
from jarvis.intelligence.memory import JarvisMemory
from jarvis.intelligence.llm import LlmClient, IntentClassifier
from jarvis.actions.parser import CommandParser
from jarvis.actions.executor import CommandExecutor
from jarvis.tts.engine import TextToSpeech

class Jarvis:
    """Production-Grade Orchestrator (Phase 2)."""

    def __init__(self, signals=None):
        self.signals = signals
        self.context = JarvisContext()
        self.memory = JarvisMemory()
        self.audio_ctrl = AudioController()
        self.gpu_mon = GpuMonitor()
        self.vad = VoiceActivityDetector()
        self.stt = SpeechToText()
        self.llm = LlmClient()
        self.tts = TextToSpeech()
        self.executor = CommandExecutor(self) # On passe JARVIS pour que l'executor accède à la mémoire

        self._pre_roll = deque(maxlen=10)
        self._speech_buffer = []
        self._silence_count = 0
        self._is_listening = False

        self.user_name = self.memory.get_user_name()
        self.history = [
            {"role": "system", "content": f"Tu es JARVIS. Concis. Utilisateur: {self.user_name}."}
        ]

    async def _audio_listener(self):
        async for frame in audio_frame_generator(self.context):
            is_speech = self.vad.is_speech(frame)
            if is_speech:
                if not self._is_listening:
                    self._is_listening = True
                    self.context.set_state(JarvisState.LISTENING)
                    self.context.trigger_stop()
                    if hasattr(self, "_resp_task") and not self._resp_task.done():
                        self._resp_task.cancel()
                    self.context.reset_stop_event()
                    self._speech_buffer = []
                    self._pre_roll.clear()
                self._speech_buffer.append(frame)
                self._silence_count = 0
            elif self._is_listening:
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
            else:
                self._pre_roll.append(frame)

    async def run(self):
        log.info(f"🚀 JARVIS V5.1 Production Ready.")

        def audio_cb(outdata, frames, time, status):
            try:
                data = self.context.playback_sync_queue.get_nowait()
                outdata[:, 0] = data
            except queue.Empty: outdata.fill(0)

        stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=audio_cb, blocksize=FRAME_SIZE)
        stream.start()

        async def playback_manager():
            while True:
                try:
                    pcm_phrase = await self.context.audio_output_queue.get()
                    if pcm_phrase is None: break
                    for i in range(0, len(pcm_phrase), FRAME_SIZE):
                        if self.context.stop_event.is_set(): break
                        chunk = pcm_phrase[i:i+FRAME_SIZE]
                        if len(chunk) < FRAME_SIZE:
                            chunk = np.concatenate([chunk, np.zeros(FRAME_SIZE-len(chunk), dtype=np.int16)])
                        self.context.playback_sync_queue.put(chunk, timeout=0.1)
                    self.context.audio_output_queue.task_done()
                    if self.context.audio_output_queue.empty(): self.context.is_speaking = False
                except asyncio.CancelledError: break
                except Exception as e: log.error(f"Playback Error: {e}")

        pb_task = asyncio.create_task(playback_manager())
        await self.tts.speak(f"Bonjour {self.user_name}.", self.context)

        try:
            async for text in self._audio_listener():
                self.context.reset_stop_event()
                self._resp_task = asyncio.create_task(self._process(text))
        finally:
            stream.stop(); stream.close()
            pb_task.cancel()
            await self.llm.close()

    async def _process(self, text):
        self.context.set_state(JarvisState.THINKING)
        if self.signals: self.signals.thinking_state_changed.emit(True)
        self.history.append({"role": "user", "content": text})

        full_resp = ""
        current_sentence = ""
        endings = re.compile(r'(?<=[.!?])\s+')

        try:
            async for token in self.llm.generate_stream(self.history):
                if self.context.stop_event.is_set(): return

                current_sentence += token
                full_resp += token

                if "]" in token:
                    for tag in CommandParser.extract_all(full_resp):
                        n, a = CommandParser.parse_call(tag)
                        self.context.set_state(JarvisState.EXECUTING)

                        # Phase 2: Orchestrator is now thin. It delegates 100% to executor.
                        res = self.executor.execute(n, a)

                        # Handle async tool responses
                        if res == "HANDLED_ASYNC_HA":
                            import os
                            asyncio.create_task(self._ha_control(a[0], a[1]))
                            res = "SUCCESS: Home Assistant command triggered."
                        elif res == "HANDLED_ASYNC_VISION":
                            asyncio.create_task(self._screenshot_and_analyze())
                            res = "SUCCESS: Vision analysis in progress."

                        self.history.append({"role": "system", "content": f"TOOL_RESULT: {res}"})
                        current_sentence = current_sentence.replace(tag, "")
                        full_resp = full_resp.replace(tag, "")
                        self.context.set_state(JarvisState.THINKING)

                if any(c in token for c in ".!?"):
                    parts = endings.split(current_sentence)
                    if len(parts) > 1:
                        for i in range(len(parts)-1):
                            s = re.sub(r"\[CMD:.*?\]", "", parts[i]).strip()
                            if s and not self.context.stop_event.is_set():
                                self.context.set_state(JarvisState.SPEAKING)
                                await self.tts.speak(s, self.context)
                        current_sentence = parts[-1]

            if current_sentence.strip() and not self.context.stop_event.is_set():
                await self.tts.speak(current_sentence.strip(), self.context)

            self.history.append({"role": "assistant", "content": full_resp})
            self.context.set_state(JarvisState.IDLE)
            if self.signals: self.signals.thinking_state_changed.emit(False)
        except Exception as e:
            log.error(f"Process Error: {e}")
            self.context.set_state(JarvisState.IDLE)

    async def _ha_control(self, entity, service):
        import aiohttp, os
        ha_url = os.getenv("HA_URL")
        ha_token = os.getenv("HA_TOKEN")
        if not ha_url or not ha_token: return
        url = f"{ha_url}/api/services/{entity.split('.')[0]}/{service}"
        headers = {"Authorization": f"Bearer {ha_token}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            await session.post(url, json={"entity_id": entity}, headers=headers)

    async def _screenshot_and_analyze(self):
        import base64, io, pyautogui, aiohttp
        from jarvis.utils.config import OLLAMA_HOST
        screenshot = pyautogui.screenshot()
        img_byte_arr = io.BytesIO()
        screenshot.save(img_byte_arr, format='PNG')
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        payload = {"model": "moondream", "prompt": "Décris brièvement cet écran.", "images": [img_base64], "stream": False}
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{OLLAMA_HOST}/api/generate", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    await self.tts.speak(f"Sur votre écran, je vois : {data.get('response', '')}", self.context)
