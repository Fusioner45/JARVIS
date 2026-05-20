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
    """Production-Grade Orchestrator (Phase 8 - Full Async)."""

    def __init__(self, signals=None):
        self.signals = signals
        self.context = JarvisContext()
        self.memory = JarvisMemory()
        self.supervisor = RuntimeSupervisor(self.context)
        self.audio_ctrl = AudioController()

        try:
            self.gpu_mon = GpuMonitor()
        except Exception:
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

    async def _set_ducking(self, duck: bool):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.audio_ctrl.set_ducking, duck)

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
                        await self._set_ducking(True)
                        with perf_tracker(audio_log, "End-to-End STT"):
                            transcript = await self.stt.transcribe(self._speech_buffer)
                        await self._set_ducking(False)

                        if transcript:
                            if self.signals: self.signals.transcription_received.emit(transcript)
                            yield transcript
                    self._speech_buffer = []
            else:
                self._pre_roll.append(frame)

    async def run(self):
        log.info(f"🚀 JARVIS V8.1 Hardened Runtime.")

        # Safety: Clear queues before start
        while not self.context.audio_output_queue.empty():
            try: self.context.audio_output_queue.get_nowait(); self.context.audio_output_queue.task_done()
            except: break
        while not self.context.playback_sync_queue.empty():
            try: self.context.playback_sync_queue.get_nowait()
            except: break

        self._running = True
        self._run_bg("supervisor", self.supervisor.start())

        def audio_cb(outdata, frames, time, status):
            try:
                data = self.context.playback_sync_queue.get_nowait()
                outdata[:, 0] = data
            except (queue.Empty, Exception): outdata.fill(0)

        stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=audio_cb, blocksize=FRAME_SIZE)
        stream.start()

        async def playback_manager():
            last_activity = time.perf_counter()
            while self._running:
                try:
                    # Watchdog: force reset is_speaking if stuck for > 5s without audio
                    if self.context.is_speaking and (time.perf_counter() - last_activity > 5.0) and self.context.audio_output_queue.empty():
                        log.warning("⚠️ Watchdog: is_speaking était bloqué. Reset forcé.")
                        self.context.is_speaking = False

                    pcm_phrase = await asyncio.wait_for(self.context.audio_output_queue.get(), timeout=1.0)
                    if pcm_phrase is None: break

                    self.context.is_speaking = True
                    last_activity = time.perf_counter()

                    for i in range(0, len(pcm_phrase), FRAME_SIZE):
                        if self.context.stop_event.is_set(): break
                        chunk = pcm_phrase[i:i+FRAME_SIZE]
                        if len(chunk) < FRAME_SIZE:
                            chunk = np.concatenate([chunk, np.zeros(FRAME_SIZE-len(chunk), dtype=np.int16)])

                        try:
                            # Increase timeout and don't silently drop chunks
                            self.context.playback_sync_queue.put(chunk, timeout=0.5)
                        except queue.Full:
                            log.warning("⚠️ Playback queue full, chunk dropped after timeout.")
                            continue

                    self.context.audio_output_queue.task_done()

                    if self.context.audio_output_queue.empty():
                        await asyncio.sleep(0.15) # Grace period
                        if self.context.audio_output_queue.empty():
                            self.context.is_speaking = False

                except (asyncio.TimeoutError, asyncio.CancelledError):
                    if self.context.audio_output_queue.empty():
                        self.context.is_speaking = False
                    continue
                except Exception as e: log.error(f"PB Error: {e}")

        pb_task = asyncio.create_task(playback_manager(), name="playback_manager")

        # UI Feedback: Notify that initialization is done
        if self.signals:
            self.signals.transcription_received.emit("SYSTÈMES EN LIGNE. EN ATTENTE...")

        self._run_bg("greeting", self.tts.speak(f"Bonjour {self.user_name}.", self.context))

        try:
            listener = self._audio_listener()
            async for text in listener:
                self.context.reset_stop_event()
                if self._resp_task and not self._resp_task.done():
                    self._resp_task.cancel()
                self._resp_task = asyncio.create_task(self._process(text), name="llm_process")
        except Exception as e:
            log.critical(f"Loop Exception: {e}")
        finally:
            self._running = False
            self.supervisor.stop()
            stream.stop(); stream.close()
            self.context.trigger_stop()
            try: await asyncio.wait_for(self.context.audio_output_queue.put(None), timeout=0.5)
            except: pass
            await pb_task
            for t in list(self._bg_tasks): t.cancel()
            await self.llm.close()

    async def _process(self, text):
        start_time = time.perf_counter()
        # History fix: Add user text immediately
        self.history.append({"role": "user", "content": text})

        try:
            self.context.set_state(JarvisState.THINKING)
            if self.signals: self.signals.thinking_state_changed.emit(True)

            memory_ctx = self.memory.get_all_context()
            temp_history = self.history[:-1] + [{"role": "system", "content": f"Context:\n{memory_ctx}"}]
            temp_history.append({"role": "user", "content": text})

            full_resp = ""
            current_sentence = ""
            endings = re.compile(r'(?<=[.!?])\s+')

            async for token in self.llm.generate_stream(temp_history):
                if self.context.stop_event.is_set(): return

                current_sentence += token
                full_resp += token

                if "]" in token and "[CMD:" in full_resp:
                    for tag in CommandParser.extract_all(full_resp):
                        n, a = CommandParser.parse_call(tag)
                        self.context.set_state(JarvisState.EXECUTING)
                        res = await self.executor.execute(n, a)

                        if res == "HANDLED_ASYNC_HA":
                            self._run_bg(f"ha_{n}", self._ha_control(a[0], a[1]))
                            res = "SUCCESS: HA triggered."
                        elif res == "HANDLED_ASYNC_VISION":
                            self._run_bg("vision", self._screenshot_and_analyze())
                            res = "SUCCESS: Vision triggered."

                        self.history.append({"role": "assistant", "content": f"[Résultat action {n}]: {res}"})
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

            self.history.append({"role": "assistant", "content": full_resp})
            if len(self.history) > 20: self.history = [self.history[0]] + self.history[-10:]

            elapsed = (time.perf_counter() - start_time) * 1000
            log.info(f"⚡ Turn Latency: {elapsed:.2f}ms")

            self.context.set_state(JarvisState.IDLE)
            if self.signals: self.signals.thinking_state_changed.emit(False)

        except asyncio.CancelledError: pass
        except Exception as e:
            log.error(f"Process Error: {e}")
            self.context.set_state(JarvisState.IDLE)

    async def _ha_control(self, entity, service):
        import aiohttp, os
        try:
            ha_url = os.getenv("HA_URL")
            ha_token = os.getenv("HA_TOKEN")
            if not ha_url or not ha_token:
                log.warning("HA: variables HA_URL/HA_TOKEN manquantes.")
                return
            url = f"{ha_url}/api/services/{entity.split('.')[0]}/{service}"
            headers = {"Authorization": f"Bearer {ha_token}", "Content-Type": "application/json"}
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json={"entity_id": entity},
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as r:
                    log.info(f"HA Response: {r.status} for {entity}/{service}")
        except asyncio.TimeoutError:
            log.error(f"HA Timeout: {entity}/{service}")
        except aiohttp.ClientError as e:
            log.error(f"HA Network Error: {e}")
        except Exception as e:
            log.error(f"HA Control Error: {e}")

    async def _screenshot_and_analyze(self):
        import base64, io, pyautogui, aiohttp
        from jarvis.utils.config import OLLAMA_HOST
        try:
            loop = asyncio.get_running_loop()
            screenshot = await loop.run_in_executor(None, pyautogui.screenshot)
            img_byte_arr = io.BytesIO()
            screenshot.save(img_byte_arr, format='PNG')
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            payload = {
                "model": "moondream",
                "prompt": "Décris précisément ce que tu vois sur cet écran.",
                "images": [img_base64],
                "stream": False
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        await self.tts.speak(
                            f"Vision : {data.get('response', 'Aucune réponse.')}", self.context
                        )
                    else:
                        log.error(f"Vision HTTP Error: {resp.status}")
        except asyncio.TimeoutError:
            log.error("Vision Timeout: Ollama moondream n'a pas répondu.")
        except Exception as e:
            log.error(f"Vision Error: {e}")
