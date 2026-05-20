import requests

payload = {
    "model": "llama3",
    "prompt": "Dis bonjour",
    "stream": False
}

r = requests.post(
    "http://localhost:11434/api/generate",
    json=payload
)

print(r.json()["response"])