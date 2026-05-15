import logging
import sys
import os

def setup_logger(name: str, log_file: str, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt="%H:%M:%S")

    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)

    handler = logging.FileHandler(f'logs/{log_file}')
    handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger

# Separate loggers
audio_log = setup_logger("Jarvis.Audio", "audio.log")
llm_log = setup_logger("Jarvis.LLM", "llm.log")
action_log = setup_logger("Jarvis.Actions", "actions.log")
system_log = setup_logger("Jarvis.System", "system.log")
memory_log = setup_logger("Jarvis.Memory", "memory.log")
stt_log = setup_logger("Jarvis.STT", "stt.log")
tts_log = setup_logger("Jarvis.TTS", "tts.log")
