import io
import asyncio
import numpy as np
import edge_tts
from pydub import AudioSegment
from jarvis.utils.config import EDGE_TTS_VOICE, SAMPLE_RATE
from jarvis.utils.logger import audio_log as log, perf_tracker
from jarvis.core.context import JarvisContext

class TextToSpeech:
    def __init__(self):
        self.voice = EDGE_TTS_VOICE

    async def speak(self, text: str, context: JarvisContext):
        if not text: return
        log.info(f"🎙️ TTS : {text[:50]}...")
        context.is_speaking = True

        try:
            with perf_tracker(log, "TTS Generation"):
                communicate = edge_tts.Communicate(text, self.voice)
                mp3_data = io.BytesIO()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio": mp3_data.write(chunk["data"])

                if mp3_data.tell() == 0:
                    context.is_speaking = False
                    return

                mp3_data.seek(0)
                audio = AudioSegment.from_file(mp3_data, format="mp3")
                audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
                pcm = np.frombuffer(audio.raw_data, dtype=np.int16)

                await context.audio_output_queue.put(pcm)
        except Exception as e:
            log.error(f"TTS Error: {e}")
            context.is_speaking = False
