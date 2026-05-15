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
    """Production-Grade Orchestrator with Barge-in, ActionGuard and Feedback-loop."""

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
        self.executor = CommandExecutor()

        self._pre_roll = deque(maxlen=10)
        self._speech_buffer = []
        self._silence_count = 0
        self._is_listening = False

        self.user_name = self.memory.get_user_name()
        # Initial prompt lock (French, Role, Memory)
        self.history = [
            {"role": "system", "content": f"""Tu es JARVIS, assistant de {self.user_name}.
Tu réponds en Français, de manière ultra-concise.
L'utilisateur s'appelle {self.user_name}.

PROTOCOLE ACTIONS :
- Si une commande est requise, utilise [CMD: NOM_ACTION('arg')].
- Si une commande a échoué précédemment (FAILED), analyse l'erreur et tente une alternative.

MÉMOIRE : {self.memory.get_all_context()}
"""}
        ]

    async def _audio_listener(self):
        async for frame in audio_frame_generator(self.context):
            is_speech = self.vad.is_speech(frame)
            if is_speech:
                if not self._is_listening:
                    self._is_listening = True
                    self.context.set_state(JarvisState.LISTENING)

                    # 🛑 STOP ATOMIQUE (Barge-in instantané)
                    self.context.trigger_stop()
                    if hasattr(self, "_resp_task") and not self._resp_task.done():
                        self._resp_task.cancel()

                    self.context.reset_stop_event()
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
            else:
                self._pre_roll.append(frame)

    async def run(self):
        log.info(f"🚀 JARVIS Production V5.1 (RTX 3070 Ti) Ready.")

        # sounddevice callback (Sync context)
        def audio_cb(outdata, frames, time, status):
            try:
                data = self.context.playback_sync_queue.get_nowait()
                outdata[:, 0] = data
            except queue.Empty:
                outdata.fill(0)

        stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            callback=audio_cb, blocksize=FRAME_SIZE
        )
        stream.start()

        # Playback worker (Producer for sounddevice)
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
                    if self.context.audio_output_queue.empty():
                        self.context.is_speaking = False
                except asyncio.CancelledError: break
                except Exception as e: log.error(f"Playback Error: {e}")

        pb_task = asyncio.create_task(playback_manager())
        await self.tts.speak(f"Bonjour {self.user_name}. Prêt à vous aider.", self.context)

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

                # Action Interceptor
                if "]" in token:
                    for tag in CommandParser.extract_all(full_resp):
                        n, a = CommandParser.parse_call(tag)
                        self.context.set_state(JarvisState.EXECUTING)

                        # Command Feedback Loop
                        res = self.executor.execute(n, a)
                        if "FAILED" in res and n == "SAVE_FACT":
                            self.memory.save_memory(a[0], a[1])
                            res = "SUCCESS: Fact stored in memory."

                        self.history.append({"role": "system", "content": f"ACTION_FEEDBACK: {res}"})
                        sentence = sentence.replace(tag, "") if 'sentence' in locals() else ""
                        current_sentence = current_sentence.replace(tag, "")
                        full_resp = full_resp.replace(tag, "")
                        self.context.set_state(JarvisState.THINKING)

                # Sentence streaming for TTS
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
