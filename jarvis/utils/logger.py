import logging
import sys
import os
import time
from contextlib import contextmanager

def setup_logger(name: str, log_file: str, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt="%H:%M:%S")

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

# Domain-specific loggers
audio_log = setup_logger("Jarvis.Audio", "audio.log")
llm_log = setup_logger("Jarvis.LLM", "llm.log")
action_log = setup_logger("Jarvis.Actions", "actions.log")
system_log = setup_logger("Jarvis.System", "system.log")

@contextmanager
def perf_tracker(domain_logger, label: str):
    start = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - start) * 1000
    domain_logger.info(f"⏱️ PERF | {label}: {elapsed:.2f}ms")
