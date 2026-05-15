import asyncio
import json
import aiohttp
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
        payload = {"model": self.model, "messages": messages, "temperature": 0.7, "stream": True}

        first_token = True
        try:
            async with self.session.post(url, json=payload, timeout=120) as resp:
                resp.raise_for_status()
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
                                    log.info("🚀 LLM : Premier token reçu.")
                                    first_token = False
                                yield token
                        except: continue
        except Exception as e:
            log.error(f"LLM Stream Error: {e}")
            yield "Erreur de connexion LLM."

class IntentClassifier:
    @staticmethod
    def classify(text: str) -> JarvisState:
        text = text.lower()
        actions = ["ouvre", "lance", "cherche", "musique", "température", "mémorise", "rappelle", "indexe"]
        if any(kw in text for kw in actions):
            return JarvisState.EXECUTING
        return JarvisState.THINKING
