import asyncio
from jarvis.tts.engine import TextToSpeech
from jarvis.core.context import JarvisContext

async def main():
    tts = TextToSpeech()
    context = JarvisContext()

    await tts.speak(
        "Bonjour Fusion, système opérationnel.",
        context
    )

asyncio.run(main())