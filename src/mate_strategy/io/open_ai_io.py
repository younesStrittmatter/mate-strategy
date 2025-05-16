import time, json, re

from openai import OpenAI

from src.mate_strategy._config import config


def ask_ai(prompt: str,
           model: str = None,
           temperature: float = 0.2,
           max_tokens: int = 1024,
           system: str | None = None) -> str:
    """
    Send a single‑turn prompt and return the assistant's plaintext reply.
    Centralised here so every Step can `from …utils.llm import ask_ai`.
    """
    if model is None:
        model = config["assist-model"]

    client = OpenAI(api_key=config["openai-key"])
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    time.sleep(config.get("query-sleep", 0))  # polite rate limiting
    return resp.choices[0].message.content.strip()
