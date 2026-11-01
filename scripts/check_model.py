"""B-01 check: one real call to the model provider must return text.
Usage: python scripts/check_model.py"""
import sys

import requests

sys.path.insert(0, ".")
from src import config  # noqa: E402

if not config.OPENROUTER_API_KEY or config.OPENROUTER_API_KEY == "your_key_here":
    sys.exit("OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key.")

resp = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"},
    json={"model": config.MODEL_NAME, "messages": [{"role": "user", "content": "Reply with the word ready."}]},
    timeout=30,
)
print(resp.status_code)
resp.raise_for_status()
print(resp.json()["choices"][0]["message"]["content"])
