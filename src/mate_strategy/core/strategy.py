from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any, Protocol
import time


class Strategy(Protocol):
    def __call__(self, prompt: str) -> Any: ...


# ----------------------------------------------------------------------
@dataclass
class SimpleCall(Strategy):
    ask_ai:     Callable[[str], str]
    accept:     Callable[[Any], bool]

    def __call__(self, prompt: str) -> Any:
        out_raw = self.ask_ai(prompt)
        if not self.accept(out):
            raise ValueError("validation failed")
        return out


# ----------------------------------------------------------------------
@dataclass
class Retry(Strategy):
    inner: Strategy
    retries: int = 2
    backoff: float = 0.3

    def __call__(self, prompt: str) -> Any:
        for i in range(self.retries + 1):
            try:
                return self.inner(prompt)
            except Exception as err:
                if i == self.retries:
                    raise
                time.sleep(self.backoff)


# ----------------------------------------------------------------------
@dataclass
class WithFallback(Strategy):
    primary: Strategy
    fallback: Strategy

    def __call__(self, prompt: str) -> Any:
        try:
            return self.primary(prompt)
        except Exception:
            return self.fallback(prompt)
