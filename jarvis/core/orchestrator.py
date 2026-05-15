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

        self._pre_roll = deque(maxlen=10)
        self._speech_buffer = []
        self._silence_count = 0
        self._is_listening = False
        self.playback_queue = queue.Queue(maxsize=100)

        self.history = [{"role": "system", "content": f"Tu es JARVIS. Concis. Utilisateur: {self.memory.get_user_name()}. Contexte: {self.memory.get_all_context()}"}]

    async def _audio_listener(self):
        async for frame in audio_frame_generator(self.context):
            is_speech = self.vad.is_speech(frame)
            if is_speech:
                if not self._is_listening:
                    self._is_listening = True
                    self.context.set_state(JarvisState.LISTENING)
                    if hasattr(self, "_resp_task") and not self._resp_task.done(): self._resp_task.cancel()
                    self._clear_playback()
                    self._speech_buffer = list(self._pre_roll)
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
            else: self._pre_roll.append(frame)

    async def run(self):
        log.info(f"🚀 JARVIS V5 Modulaire Démarré.")

        def audio_cb(outdata, frames, time, status):
            try:
                data = self.playback_queue.get_nowait()
                outdata[:, 0] = data
            except queue.Empty: outdata.fill(0)

        stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=audio_cb, blocksize=FRAME_SIZE)
        stream.start()

        async def playback_manager():
            while True:
                chunk = await self.context.audio_output_queue.get()
                if chunk is None: break
                for i in range(0, len(chunk), FRAME_SIZE):
                    seg = chunk[i:i+FRAME_SIZE]
                    if len(seg) < FRAME_SIZE: seg = np.concatenate([seg, np.zeros(FRAME_SIZE-len(seg), dtype=np.int16)])
                    self.playback_queue.put(seg)
                if self.context.audio_output_queue.empty(): self.context.is_speaking = False

        pb_task = asyncio.create_task(playback_manager())
        await self.tts.speak(f"Bonjour {self.memory.get_user_name()}.", self.context)

        try:
            async for text in self._audio_listener():
                self._resp_task = asyncio.create_task(self._process(text))
                await self._resp_task
        finally:
            stream.stop(); stream.close()
            await self.context.audio_output_queue.put(None); await pb_task
            await self.llm.close()

    def _clear_playback(self):
        while not self.playback_queue.empty():
            try: self.playback_queue.get_nowait()
            except: break
        self.context.is_speaking = False

    async def _process(self, text):
        self.context.set_state(JarvisState.THINKING)
        if self.signals: self.signals.thinking_state_changed.emit(True)
        self.history.append({"role": "user", "content": text})

        resp = ""
        sentence = ""
        endings = re.compile(r'(?<=[.!?])\s+')

        try:
            async for token in self.llm.generate_stream(self.history):
                sentence += token
                resp += token
                if "]" in token:
                    for tag in CommandParser.extract_all(resp):
                        n, a = CommandParser.parse_call(tag)
                        self.context.set_state(JarvisState.EXECUTING)
                        res = CommandExecutor.execute(n, a, self)
                        if res == "HANDLED_BY_ORCHESTRATOR":
                            if n == "SAVE_FACT": self.memory.save_memory(a[0], a[1]); res = "SUCCESS"
                            elif n == "INDEX_PDF": self.memory.index_pdf(a[0]); res = "SUCCESS"
                        self.history.append({"role": "system", "content": f"Result: {res}"})
                        sentence = sentence.replace(tag, ""); resp = resp.replace(tag, "")
                        self.context.set_state(JarvisState.THINKING)

                if any(c in token for c in ".!?"):
                    parts = endings.split(sentence)
                    if len(parts) > 1:
                        for i in range(len(parts)-1):
                            s = re.sub(r"\[CMD:.*?\]", "", parts[i]).strip()
                            if s:
                                self.context.set_state(JarvisState.SPEAKING)
                                await self.tts.speak(s, self.context)
                        sentence = parts[-1]

            if sentence.strip():
                s = re.sub(r"\[CMD:.*?\]", "", sentence).strip()
                if s: await self.tts.speak(s, self.context)

            self.history.append({"role": "assistant", "content": resp})
            self.context.set_state(JarvisState.IDLE)
            if self.signals: self.signals.thinking_state_changed.emit(False)
        except Exception as e: log.error(f"Orchestrator Error: {e}")
