import asyncio
import json
import aiohttp
from typing import AsyncGenerator, List, Dict
from jarvis.utils.config import OLLAMA_HOST, OLLAMA_MODEL
from jarvis.utils.logger import llm_log as log
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
        if self.session and not self.session.closed:
            await self.session.close()

    async def generate_stream(self, messages: List[Dict]) -> AsyncGenerator[str, None]:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 512,
            "stream": True,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                await self._ensure_session()
                async with self.session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(connect=5, total=120)
                ) as resp:
                    if resp.status == 500:
                        log.warning(f"Ollama 500 (Essai {attempt+1})")
                        await asyncio.sleep(1)
                        continue

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
                                if token: yield token
                            except: continue
                    return
            except Exception as e:
                log.error(f"Erreur Ollama : {e}")
                if attempt == max_retries - 1:
                    yield "Je rencontre une difficulté technique avec mon cerveau local."
                else:
                    await asyncio.sleep(1)

class IntentClassifier:
    @staticmethod
    def classify(text: str) -> JarvisState:
        text = text.lower()
        # Mots-clés déclencheurs d'action
        action_keywords = ["ouvre", "lance", "cherche", "musique", "température", "mémorise", "rappelle", "indexe"]
        if any(kw in text for kw in action_keywords):
            return JarvisState.EXECUTING
        return JarvisState.THINKING
