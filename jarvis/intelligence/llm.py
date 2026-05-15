import asyncio
import json
import aiohttp
import time
from typing import AsyncGenerator, List, Dict
from jarvis.utils.config import OLLAMA_HOST, OLLAMA_MODEL
from jarvis.utils.logger import llm_log as log, perf_tracker
from jarvis.core.states import JarvisState

class LlmClient:
    def __init__(self, base_url: str = OLLAMA_HOST, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session: aiohttp.ClientSession | None = None

    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session: await self.session.close()

    async def generate_stream(self, messages: List[Dict]) -> AsyncGenerator[str, None]:
        await self._ensure_session()
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.4, # Réduit pour plus de déterminisme
            "stream": True,
            "options": {
                "num_ctx": 4096,
                "num_predict": 512
            }
        }

        start_time = time.perf_counter()
        first_token = True

        try:
            async with self.session.post(url, json=payload, timeout=60) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    log.error(f"Ollama Error ({resp.status}): {error_text}")
                    yield f"Erreur LLM ({resp.status})."
                    return

                async for line in resp.content:
                    if not line: continue
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: "):
                        data_content = line_str[6:]
                        if data_content == "[DONE]": break
                        try:
                            data = json.loads(data_content)
                            token = data["choices"][0]["delta"].get("content", "")
                            if token:
                                if first_token:
                                    elapsed = (time.perf_counter() - start_time) * 1000
                                    log.info(f"🚀 LLM TTFT : {elapsed:.2f}ms")
                                    first_token = False
                                yield token
                        except: continue
        except asyncio.TimeoutError:
            log.error("LLM Timeout après 60s")
            yield "Le serveur LLM ne répond pas (Timeout)."
        except Exception as e:
            log.error(f"LLM Stream Error: {e}")
            yield "Désolé, j'ai une erreur de connexion au cerveau local."
