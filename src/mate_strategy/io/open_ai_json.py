import re
import json

from mate_strategy.io.open_ai_io import ask_ai


def ask_ai_json(prompt: str,
                model: str = None,
                temperature: float = 0.2,
                max_tokens: int = 1024,
                ) -> dict:
    """
    Send a single‑turn prompt and return the assistant's JSON reply.
    Centralised here so every Step can `from …utils.llm import ask_ai_json`.
    """
    resp = ask_ai(prompt, model=model, temperature=temperature,
                  max_tokens=max_tokens, system=SYSTEM_JSON)
    return process_to_json(resp)


def process_to_json(resp: str) -> dict:
    if not resp:
        print("Empty response")
        return {}
    resp = resp.strip()  # trim whitespace
    resp = re.sub(r",(\s*[\]}])", r"\1", resp)
    resp = re.sub(r",\s*,+", ",", resp)
    if resp.startswith("```"):
        resp = resp.strip("`")
        if resp.lstrip().lower().startswith("json"):
            resp = resp.split("\n", 1)[1]
    if not resp:
        print("Empty response after stripping")
        return {}
    try:
        return json.loads(resp)
    except json.decoder.JSONDecodeError:
        print(f"Bad JSON: {resp!r}")
        return {}


SYSTEM_JSON = "You are a JSON-only extraction assistant. Reply ONLY with valid JSON."
