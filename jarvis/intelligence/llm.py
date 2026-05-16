import asyncio
import json
import aiohttp
import time
from typing import AsyncGenerator, List, Dict
from jarvis.utils.config import OLLAMA_HOST, OLLAMA_MODEL
from jarvis.utils.logger import llm_log as log

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
            "temperature": 0.4,
            "stream": True,
            "options": {"num_ctx": 4096}
        }

        start_time = time.perf_counter()
        first_token = True
        has_yielded = False

        try:
            async with self.session.post(url, json=payload, timeout=60) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    log.error(f"Ollama Error ({resp.status}): {err}")
                    yield "Erreur Ollama."
                    return

                async for line in resp.content:
                    if not line: continue
                    line_str = line.decode("utf-8").strip()

                    if not line_str.startswith("data: "): continue

                    data_content = line_str[6:]
                    if data_content == "[DONE]": break

                    try:
                        data = json.loads(data_content)
                        token = data["choices"][0]["delta"].get("content", "")
                        if token:
                            if first_token:
                                log.info(f"🚀 LLM TTFT: {(time.perf_counter()-start_time)*1000:.2f}ms")
                                first_token = False
                            has_yielded = True
                            yield token
                    except Exception as e:
                        log.error(f"Token Error: {e}")

        except asyncio.TimeoutError:
            log.error("Ollama Timeout.")
            # Restore vocal feedback on error
            yield "Délai d'attente dépassé. Ollama ne répond pas."
        except Exception as e:
            log.error(f"LLM Connection Error: {e}")
            # Restore vocal feedback on error
            yield "Erreur de connexion au modèle de langage."
        finally:
            if not has_yielded:
                log.warning("Ollama stream ended without yielding any tokens.")
