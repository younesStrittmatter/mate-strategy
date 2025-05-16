from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Any, Protocol
import time, copy

from src.mate_strategy.json.open_ai_io import ask_ai_json
from src.mate_strategy.json.prompt import Prompt
from src.mate_strategy.json.key import Key
from src.mate_strategy.json.auto_rule import AutoRule
from src.mate_strategy.json.rules import Interval


class Strategy(Protocol):
    def __call__(self, prompt: str) -> Any: ...


CONF_KEY = Key("confidence", int, Interval(0, 100),
               desc="How confident are you in your answer?")


def _clone_with_conf(prompt: Prompt) -> Prompt:
    """Return a *new* Prompt whose Response includes the confidence key."""
    resp = prompt.response
    if any(k.name == "confidence" for k in resp.keys):
        return prompt  # already present
    new_resp = replace(resp, keys=resp.keys + (CONF_KEY,))
    return Prompt(prompt.prompt, new_resp)


# ----------------------------------------------------------------------
@dataclass
class SimpleCall(Strategy):
    """
    Mini-strategy: single LLM call, plus schema validation.

    Examples
    --------
    >>> from mate_strategy.json.response import Response
    >>> from mate_strategy.json.key      import Key
    >>> schema = Response([Key("x", int, Interval(0, 10))])
    >>> prompt = Prompt("fill", schema)

    Fake LLM that returns a valid dict
    >>> ok_ai   = lambda *_: {"x": 3}
    >>> call_ok = SimpleCall(ask_ai=ok_ai)
    >>> call_ok(prompt)['x']
    3

    Fake LLM that returns a valid dict
    >>> bad_ai   = lambda *_: {"x": 11}
    >>> call_ok = SimpleCall(ask_ai=bad_ai)
    >>> call_ok(prompt)['x']
    Traceback (most recent call last):
      ...
    ValueError: validation failed

    We can also use a prompt with placeholders:
    >>> prompt = Prompt("fill {x} with a random value", schema)
    >>> ok_ai   = lambda *_: {"x": 7}
    >>> call_ok = SimpleCall(ask_ai=ok_ai)
    >>> call_ok(prompt, x=42)['x']
    7

    We still run the call but warn if placeholers are not used correctly:
    >>> call_ok(prompt, y=32)['x']
    [WARN] Key not found in prompt: y
    7
    """
    ask_ai: Callable[[str], dict] = ask_ai_json
    accept: Callable[[Any], bool] = lambda x: True

    def __call__(self, prompt: Prompt, log_prompt=False, **kwargs) -> Any:
        if log_prompt:
            print(prompt(**kwargs))
        out_raw = self.ask_ai(prompt(**kwargs))
        if not self.accept(out_raw) or not prompt.validate(out_raw):
            raise ValueError("validation failed")
        return out_raw


@dataclass
class SimpleConfidenceCall(Strategy):
    ask_ai: Callable[[str], dict] = ask_ai_json
    accept: Callable[[dict], bool] = lambda x: True
    threshold: int = 85

    def __call__(self, prompt: Prompt, **kwargs) -> Any:
        prompt_with_conf = _clone_with_conf(prompt)
        out_raw = self.ask_ai(prompt_with_conf(**kwargs))
        if (not self.accept(out_raw) or
                not prompt_with_conf.validate(out_raw) or
                out_raw["confidence"] < self.threshold):
            raise ValueError("validation failed")
        return out_raw


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


@dataclass
class HealFields:
    """
    Wrap another Strategy.
    If the first call raises *validation* errors coming from an AutoRule
    (i.e. invalid nested object), we re-ask **only** that sub-JSON,
    patch it in place, and repeat recursively.
    """
    inner: Strategy
    retries: int = 2  # per field
    with_parent: bool = False

    # ------------------------------------------------------------------
    def __call__(self, prompt: Prompt, **kw) -> Any:
        raw = self.inner(prompt, **kw)  # may raise ValueError
        try:
            prompt.validate(raw)
            return raw  # all good
        except Exception as err:
            # find first failing key that is an AutoRule
            fixed = copy.deepcopy(raw)
            if self._repair_recursive(prompt.response.keys, fixed, prompt):
                return fixed
            raise  # bubble up original error

    # ------------------------------------------------------------------
    def _repair_recursive(self,
                          keys: tuple[Key, ...],
                          data: dict,
                          parent_prompt: Prompt) -> bool:
        """
        Walk keys; try to repair the first failing AutoRule object.
        Returns True on success.
        """
        for k in keys:
            if k.name not in data:
                return False
            val = data[k.name]
            rule = k.rule
            # plain key – nothing to repair deeper
            if not isinstance(rule, AutoRule):
                continue

            # if already valid, descend deeper if Parseable contains schema
            if rule(val):
                # recurse into children schema if available
                if isinstance(val, dict):
                    ok = self._repair_recursive(rule.cls.__schema__,
                                                val, parent_prompt)
                    if ok:
                        return True
                continue  # already fine, keep scanning

            # ---- we found a bad sub-object → try healing -------------
            for _ in range(self.retries):
                repair_p = rule.repair_prompt(val)
                if self.with_parent:
                    repair_p = (
                            "----- ORIGINAL PROMPT CONTEXT -----\n"
                            f"{parent_prompt()}\n"
                            "----------- FIX THIS -------------\n"
                            + repair_p
                    )
                sub_prompt = Prompt(repair_p, parent_prompt.response)
                try:
                    fixed_sub = self.inner(sub_prompt)  # ask LLM only for sub-json
                    if rule(fixed_sub):
                        data[k.name] = fixed_sub  # patch
                        return True
                except Exception:
                    continue
            return False
        return False
