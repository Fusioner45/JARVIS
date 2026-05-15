import io
import asyncio
import numpy as np
import edge_tts
from pydub import AudioSegment
from jarvis.utils.config import EDGE_TTS_VOICE, SAMPLE_RATE
from jarvis.utils.logger import tts_log as log
from jarvis.core.context import JarvisContext

class VoiceMorpher:
    @staticmethod
    def apply_robot_filter(input_pcm: np.ndarray, sample_rate: int) -> np.ndarray:
        import tempfile
        import soundfile as sf
        import os
        import subprocess

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fin, \
                 tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fout:
                sf.write(fin.name, input_pcm, sample_rate)
                fin.close()
                fout.close()

                cmd = [
                    'ffmpeg', '-y', '-i', fin.name,
                    '-af', 'asetrate=16000*0.8,atempo=1.25,vibrato=f=10:d=0.5',
                    fout.name
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STNULL)

                out_data, _ = sf.read(fout.name, dtype='int16')
                os.unlink(fin.name)
                os.unlink(fout.name)
                return out_data
        except Exception as e:
            log.error(f"Erreur VoiceMorpher : {e}")
            return input_pcm

class TextToSpeech:
    def __init__(self, voice: str = EDGE_TTS_VOICE):
        self.voice = voice
        self.use_morphing = False

    async def speak(self, text: str, context: JarvisContext):
        """Lit le texte et injecte les chunks PCM dans la file d'attente du contexte."""
        if not text: return

        log.info(f"🎙️ Synthèse vocale : {text[:50]}...")
        context.is_speaking = True

        try:
            communicate = edge_tts.Communicate(text, voice=self.voice)
            mp3_data = io.BytesIO()

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_data.write(chunk["data"])

            if mp3_data.tell() == 0:
                context.is_speaking = False
                return

            mp3_data.seek(0)
            loop = asyncio.get_running_loop()

            def _decode():
                try:
                    audio_segment = AudioSegment.from_file(mp3_data, format="mp3")
                    audio_segment = audio_segment.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
                    return np.frombuffer(audio_segment.raw_data, dtype=np.int16)
                except Exception as e:
                    log.error(f"Erreur décodage MP3 : {e}")
                    return None

            pcm_array = await loop.run_in_executor(None, _decode)

            if pcm_array is not None:
                if self.use_morphing:
                    pcm_array = await loop.run_in_executor(None, VoiceMorpher.apply_robot_filter, pcm_array, SAMPLE_RATE)

                await context.audio_output_queue.put(pcm_array)
            else:
                context.is_speaking = False

        except Exception as e:
            log.error(f"Erreur TTS : {e}")
            context.is_speaking = False
