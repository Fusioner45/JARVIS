import asyncio
import time
import re
import queue
import numpy as np
import sounddevice as sd
from collections import deque
from jarvis.utils.config import SAMPLE_RATE, FRAME_SIZE, SILENCE_FRAMES_THRESHOLD
from jarvis.utils.logger import system_log as log, perf_tracker, audio_log
from jarvis.utils.helpers import AudioController, GpuMonitor
from jarvis.utils.monitor import RuntimeSupervisor
from jarvis.core.context import JarvisContext
from jarvis.core.states import JarvisState
from jarvis.perception.audio import audio_frame_generator, VoiceActivityDetector
from jarvis.perception.stt import SpeechToText
from jarvis.intelligence.memory import JarvisMemory
from jarvis.intelligence.llm import LlmClient
from jarvis.actions.parser import CommandParser
from jarvis.actions.executor import CommandExecutor
from jarvis.tts.engine import TextToSpeech

class Jarvis:
    """Production-Grade Orchestrator (Phase 7 - Hardened Runtime)."""

    def __init__(self, signals=None):
        self.signals = signals
        self.context = JarvisContext()
        self.memory = JarvisMemory()
        self.supervisor = RuntimeSupervisor(self.context)
        self.audio_ctrl = AudioController()

        try:
            self.gpu_mon = GpuMonitor()
        except Exception as e:
            log.warning(f"GpuMonitor Init Failed: {e}")
            self.gpu_mon = None

        self.vad = VoiceActivityDetector()
        self.stt = SpeechToText()
        self.llm = LlmClient()
        self.tts = TextToSpeech()
        self.executor = CommandExecutor(self)

        self._pre_roll = deque(maxlen=10)
        self._speech_buffer = []
        self._silence_count = 0
        self._is_listening = False
        self._resp_task = None
        self._bg_tasks = set()
        self._running = False

        self.user_name = self.memory.get_user_name()
        self.history = [
            {"role": "system", "content": f"Tu es JARVIS. Concis. Utilisateur: {self.user_name}."}
        ]

    def _run_bg(self, name, coro):
        task = asyncio.create_task(coro, name=name)
        self._bg_tasks.add(task)
        self.supervisor.track_task(name, task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _audio_listener(self):
        async for frame in audio_frame_generator(self.context):
            if not self._running: break
            is_speech = self.vad.is_speech(frame)
            if is_speech:
                if not self._is_listening:
                    self._is_listening = True
                    self.context.set_state(JarvisState.LISTENING)
                    self.context.trigger_stop()

                    if self._resp_task and not self._resp_task.done():
                        self._resp_task.cancel()

                    self.context.reset_stop_event()
                    self._speech_buffer = []
                    self._speech_buffer.extend(list(self._pre_roll))
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
                        with perf_tracker(audio_log, "End-to-End Transcription"):
                            transcript = await self.stt.transcribe(self._speech_buffer)
                        self.audio_ctrl.set_ducking(False)

                        if transcript:
                            if self.signals: self.signals.transcription_received.emit(transcript)
                            yield transcript
                    self._speech_buffer = []
            else:
                self._pre_roll.append(frame)

    async def run(self):
        log.info(f"🚀 JARVIS V7.0 Hardened Runtime.")
        self._running = True

        # Start Supervisor
        self._run_bg("supervisor", self.supervisor.start())

        def audio_cb(outdata, frames, time, status):
            if status: log.warning(f"Audio Out Status: {status}")
            try:
                data = self.context.playback_sync_queue.get_nowait()
                outdata[:, 0] = data
            except queue.Empty: outdata.fill(0)
            except Exception: pass

        stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=audio_cb, blocksize=FRAME_SIZE)
        stream.start()

        async def playback_manager():
            while self._running:
                try:
                    pcm_phrase = await asyncio.wait_for(self.context.audio_output_queue.get(), timeout=1.0)
                    if pcm_phrase is None:
                        self.context.audio_output_queue.task_done()
                        break

                    for i in range(0, len(pcm_phrase), FRAME_SIZE):
                        if self.context.stop_event.is_set(): break
                        chunk = pcm_phrase[i:i+FRAME_SIZE]
                        if len(chunk) < FRAME_SIZE:
                            chunk = np.concatenate([chunk, np.zeros(FRAME_SIZE-len(chunk), dtype=np.int16)])

                        try:
                            self.context.playback_sync_queue.put(chunk, timeout=0.1)
                        except queue.Full:
                            continue

                    self.context.audio_output_queue.task_done()
                    if self.context.audio_output_queue.empty(): self.context.is_speaking = False
                except asyncio.TimeoutError: continue
                except asyncio.CancelledError: break
                except Exception as e: log.error(f"Playback Error: {e}")

        pb_task = asyncio.create_task(playback_manager())
        self._run_bg("initial_greeting", self.tts.speak(f"Bonjour {self.user_name}.", self.context))

        try:
            async for text in self._audio_listener():
                self.context.reset_stop_event()
                if self._resp_task and not self._resp_task.done():
                    self._resp_task.cancel()
                self._resp_task = asyncio.create_task(self._process(text), name="llm_process")
        except Exception as e:
            log.critical(f"Main Loop Exception: {e}")
        finally:
            self._running = False
            self.supervisor.stop()
            log.info("Cleaning up resources...")
            stream.stop(); stream.close()
            self.context.trigger_stop()
            try:
                await asyncio.wait_for(self.context.audio_output_queue.put(None), timeout=1.0)
            except: pass
            await pb_task
            for task in list(self._bg_tasks):
                log.debug(f"Cancelling task: {task.get_name()}")
                task.cancel()
            await self.llm.close()
            log.info("Shutdown complete.")

    async def _process(self, text):
        start_time = time.perf_counter()
        try:
            self.context.set_state(JarvisState.THINKING)
            if self.signals: self.signals.thinking_state_changed.emit(True)

            memory_ctx = self.memory.get_all_context()
            temp_history = self.history + [{"role": "system", "content": f"Mémoire:\n{memory_ctx}"}]
            temp_history.append({"role": "user", "content": text})

            full_resp = ""
            current_sentence = ""
            endings = re.compile(r'(?<=[.!?])\s+')

            async for token in self.llm.generate_stream(temp_history):
                if self.context.stop_event.is_set():
                    log.info("LLM Generation aborted.")
                    return

                current_sentence += token
                full_resp += token

                if "]" in token:
                    for tag in CommandParser.extract_all(full_resp):
                        n, a = CommandParser.parse_call(tag)
                        self.context.set_state(JarvisState.EXECUTING)
                        res = self.executor.execute(n, a)

                        if res == "HANDLED_ASYNC_HA":
                            self._run_bg(f"ha_{n}", self._ha_control(a[0], a[1]))
                            res = "SUCCESS: HA triggered."
                        elif res == "HANDLED_ASYNC_VISION":
                            self._run_bg("vision_analysis", self._screenshot_and_analyze())
                            res = "SUCCESS: Vision triggered."

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
                self.context.set_state(JarvisState.SPEAKING)
                await self.tts.speak(current_sentence.strip(), self.context)

            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": full_resp})

            # Pruning History (Phase 8)
            if len(self.history) > 20:
                self.history = [self.history[0]] + self.history[-10:]
                log.info("Pruned LLM History for latency control.")

            elapsed = (time.perf_counter() - start_time) * 1000
            log.info(f"⚡ Turn Latency: {elapsed:.2f}ms")

            self.context.set_state(JarvisState.IDLE)
            if self.signals: self.signals.thinking_state_changed.emit(False)

        except asyncio.CancelledError:
            log.debug("Process task canceled.")
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
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={"entity_id": entity}, headers=headers, timeout=5) as r:
                    log.info(f"HA Service Response: {r.status}")
        except Exception as e: log.error(f"HA Error: {e}")

    async def _screenshot_and_analyze(self):
        import base64, io, pyautogui, aiohttp
        from jarvis.utils.config import OLLAMA_HOST
        try:
            screenshot = pyautogui.screenshot()
            img_byte_arr = io.BytesIO()
            screenshot.save(img_byte_arr, format='PNG')
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            payload = {"model": "moondream", "prompt": "Décris l'écran.", "images": [img_base64], "stream": False}
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        await self.tts.speak(f"Vision : {data.get('response', '')}", self.context)
        except Exception as e: log.error(f"Vision Error: {e}")
