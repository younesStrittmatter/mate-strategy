"""
A minimal “lego-kit” for building and composing LLM strategies.

Public objects
==============
    • Prompt           – (template + Parseable)  → renders + validates
    • Strategy (ABC)   – single-method protocol
    • BaseStrategy     – one-shot call + retry + validation
    • Fallback         – try inner, else outer
    • kwroute          – decorator: routes   inner__attr=value
                          or   inner={"attr": value}   to parameters
    • override_self    – decorator: prompt=…, retries=… overrides
"""

from __future__ import annotations

import time, inspect, copy, contextlib, json

from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Protocol
from contextlib import contextmanager
from functools import wraps

from mate_strategy.io.open_ai_json import ask_ai_json
from mate_strategy.prompt import Prompt
from mate_strategy.rules.predefined import Interval


# ═════════════════════════ strategy protocol ══════════════════════════
class Strategy(Protocol):
    prompt: Prompt

    def __call__(self, **kw) -> Dict[str, Any]: ...


# ───────────────────────── helpers ──────────────────────────
@contextmanager
def _temp_attrs(obj, **patch):
    """
    Temporarily set attributes on *obj*; restore them afterwards.
    """
    if obj is None or not patch:
        yield
        return

    saved = {k: getattr(obj, k) for k in patch}
    try:
        for k, v in patch.items():
            setattr(obj, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(obj, k, v)


# ─────────────────── AutoRepair.__call__ fix ─────────────────
def __call__(self, *,
             inner: Strategy | None = None,
             depth: int | None = None,
             mode: str | None = None,
             prompt: Prompt | None = None,
             **tmpl):
    inner = inner or self.inner
    depth = self.depth if depth is None else depth
    mode = self.mode if mode is None else mode

    if prompt is not None:
        inner = copy.copy(inner)  # shallow clone preserves the class stack
        inner.prompt = prompt

    runner_prompt = inner.prompt
    schema = runner_prompt.schema

    reply, ok, err, exp = inner(**tmpl)
    if ok or depth == 0:
        return reply, ok, err, exp

    fixed = copy.deepcopy(reply)
    if self._repair(fixed, tmpl,
                    depth=depth,
                    mode=mode,
                    prompt=runner_prompt,
                    schema=schema):
        return fixed, True, None, None
    return reply, False, err, exp


# ───────────────────── decorator fixes ──────────────────────
def override_self(fn):
    """
    Adds two capabilities to *any* strategy that uses it:

    ①  Call-time overrides for attributes on `self` or `self.prompt`
       (exact behaviour unchanged).

    ②  **Universal logging** via an optional `log=` kwarg.
        • log=None  (default)  → no logging
        • log=print           → print debug block to stdout
        • log=logging.info    → any callable(str) works

       Extras:
        • log_pretty=True/False   → pretty JSON vs repr
        • log_tag="DBG"           → prefix in the header
    """
    sig = inspect.signature(fn)
    own_params = [p.name for p in sig.parameters.values()
                  if p.kind == p.POSITIONAL_OR_KEYWORD and p.name != "self"]

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        # ── 0. grab logging kwargs (remove from kwargs so inner fn ignores them)
        logger = kwargs.get("log") or None
        if logger is False:  # allow log=False to disable at call-site
            logger = None
        prompt_txt = None

        t0 = time.perf_counter() if logger else None

        # ── 1. *existing* patch-self / patch-prompt logic (unchanged) ──
        bound = sig.bind_partial(self, *args, **kwargs)
        bound.apply_defaults()

        self_patch, prompt_patch = {}, {}
        for name in own_params:
            if name in bound.arguments and bound.arguments[name] is not None:
                val = bound.arguments.pop(name)
                if hasattr(self, name):
                    self_patch[name] = val
                elif hasattr(getattr(self, "prompt", None), name):
                    prompt_patch[name] = val

        for k in list(bound.kwargs):
            if "__" not in k:
                continue
            root, attr = k.split("__", 1)
            (self_patch if root == "self" else prompt_patch)[attr] = bound.kwargs.pop(k)

        with _temp_attrs(self, **self_patch), \
                _temp_attrs(getattr(self, "prompt", None), **prompt_patch):
            prompt_obj = getattr(self, "prompt", None)

            if prompt_obj is not None:
                try:
                    prompt_txt = prompt_obj.render(**bound.kwargs)
                except Exception:
                    prompt_txt = f"<could-not-render {prompt_obj!r}>"

            reply, ok, err, exp = fn(*bound.args, **bound.kwargs)

        # ── 2. logging (if requested) ────────────────────────────────
        if logger:
            dt = time.perf_counter() - t0
            head = "=" * 19 + "LOG" + "=" * 19
            head += f"\n[{self.__class__.__name__}]ok={ok} Δt={dt:0.2f}s\n"
            head += "-" * 41
            if prompt_txt:
                head += f"\n[PROMPT]\n"
                head += prompt_txt
                head += '\n' + "-" * 41
            body = '[REPLY]\n'
            body += json.dumps(reply, indent=2)
            body += '\n' + "-" * 41
            msg = f"{head}\n{body}"
            if err:
                msg += f"\n  ✖ err: {err}"
            if exp:
                msg += f"\n  ✖ exp: {exp}"
            logger(msg)

        return reply, ok, err, exp

    return wrapper


def kwroute(fn: Callable) -> Callable:
    """
    Dict‑style and double‑underscore overrides for *other* keyword parameters.
    """
    sig = inspect.signature(fn)
    pnames = [p.name for p in sig.parameters.values()
              if p.kind == p.POSITIONAL_OR_KEYWORD and p.name != "self"]

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        # convert positional args to a dict keyed by param name
        bound = {name: val for name, val in zip(pnames, args)}

        patches: list[tuple[Any, dict]] = []

        # ---- direct dict overrides -------------------------
        for pname in pnames:
            if pname in kwargs and isinstance(kwargs[pname], dict):
                target = bound.setdefault(pname, getattr(self, pname))
                patches.append((target, kwargs.pop(pname)))

        # ---- double‑underscore overrides -------------------
        for key in [k for k in kwargs if "__" in k]:
            pname, attr = key.split("__", 1)
            if pname not in pnames:
                continue
            target = bound.setdefault(pname, getattr(self, pname))
            patches.append((target, {attr: kwargs.pop(key)}))

        with contextlib.ExitStack() as stack:
            for target, patch in patches:
                stack.enter_context(_temp_attrs(target, **patch))

            # positional order matters – rebuild args list
            ordered_args = [bound.get(n, getattr(self, n)) for n in pnames]
            return fn(self, *ordered_args, **kwargs)

    return wrapper


# ═══════════════════════ concrete strategies ══════════════════════════
@dataclass
class BaseStrategy(Strategy):
    """
    A simple strategy that asks and retries until a maximum number of
    `retries` is reached or the answer is valid.
    Examples:
        First, we create a prompt with a schema:
        >>> from dataclasses import dataclass
        >>> from mate_strategy.rules.predefined import Interval
        >>> from mate_strategy.schema import Schema
        >>> @dataclass
        ... class LuckyNumbers(Schema):
        ...     a: Interval[0, 10]
        ...     b: Interval[0, 10]

        Then we create a prompt with a template and the schema:
        >>> from mate_strategy.prompt import Prompt
        >>> prompt = Prompt('Give me two {attribute} numbers', LuckyNumbers)

        In this example we create a fake LLM call that returns random numbers
        (typically you can skip this step and use the default `ask_ai_json` function)
        >>> import random
        >>> def fake_ask_ai(prompt): return {'a': random.randint(0, 20), 'b': random.randint(0, 20)}

        We can now create the strategy:
        >>> base = BaseStrategy(prompt, ask_ai=fake_ask_ai)

        and then execute it ...
        >>> random.seed(42)
        >>> base(attribute='lucky', retries=8)
        ({'a': 0, 'b': 8}, True, None, None)

        ... the return value is a tuple of 4 values:
        • the answer (dict)
        • whether the answer is valid (bool)
        • the error message (if any)
        • the exception (if any)

        If the answer is invalid, it looks like this:
        >>> random.seed(0)
        >>> base(attribute='lucky', retries=0)
        ({'a': 12, 'b': 13}, False, '"a" is invalid.', '"a" must be a number between 0 and 10')

        Remember: only the first error that is found in the answer dict will be returned (lazy return).

        Allthough we have initialized the strategy with a default prompt, we can also use it with different
        prompts without reinitializing:
        >>> @dataclass
        ... class MoreLuckyNumbers(Schema):
        ...     a: Interval[0, 20]
        ...     b: Interval[0, 20]
        ...     c: Interval[0, 20]


        >>> prompt2 = Prompt('Give me the three random numbers', MoreLuckyNumbers)
        >>> base(prompt=prompt2)
        ({'a': 1, 'b': 8}, False, '"c" is missing.', '"c" must be present.')

        ... we get an invalid answer since we are now expecting three numbers.
        (ask_ai can not be overridden in the exec function, since it is not a parameter of exec).
        We can either reinitialize the strategy with the new prompt or use the `ask_ai` parameter:
        >>> base.ask_ai = lambda x: {'a': 16, 'b': 15, 'c': 15}
        >>> base(prompt=prompt2)
        ({'a': 16, 'b': 15, 'c': 15}, True, None, None)

        ... to see the exact prompt and relply, we can use the `log` parameter which acceptes a
        callable that itslef accepts a string.
        >>> reply = base(prompt=prompt2, log=print)
        ===================LOG===================
        [BaseStrategy]ok=True Δt=0.00s
        -----------------------------------------
        [PROMPT]
        Give me the three random numbers
        Fill in **valid JSON** for the fields below.
        <BLANKLINE>
        Rules
        - a must be a number between 0 and 20
        - b must be a number between 0 and 20
        - c must be a number between 0 and 20
        <BLANKLINE>
        Example:
        {
          "a": 10,
          "b": 10,
          "c": 10
        }
        <BLANKLINE>
        Return **only** the JSON object — no code-fences, no comments.
        -----------------------------------------
        [REPLY]
        {
          "a": 16,
          "b": 15,
          "c": 15
        }
        -----------------------------------------

    """
    prompt: Prompt
    ask_ai: Callable[[str], Dict[str, Any]] = ask_ai_json
    retries: int = 0
    backoff: float = 0.5

    @override_self
    def __call__(self, prompt: Prompt | None = None,
                 retries: int | None = None,
                 **tmpl):
        txt = self.prompt.render(**tmpl)
        reply, last_err, last_exp = None, None, None

        for i in range(self.retries + 1):
            reply = self.ask_ai(txt)  # ❶ call model
            ok, err, exp = self.prompt.validate(reply)  # ❷ validate JSON
            if ok:
                return reply, True, None, None
            last_err, last_exp = err, exp
            time.sleep(self.backoff)
        return reply, False, last_err, last_exp


@dataclass
class Fallback(Strategy):
    """
    A strategy that tries to call the inner strategy first and, if it fails,
    falls back to the fallback strategy.

    Example:
        We start by defining a schema and a prompt as before ...
        >>> from dataclasses import dataclass
        >>> from mate_strategy.rules.predefined import Interval
        >>> from mate_strategy.schema import Schema
        >>> @dataclass
        ... class LuckyNumbers(Schema):
        ...     a: Interval[0, 10]
        ...     b: Interval[0, 10]

        >>> from mate_strategy.prompt import Prompt
        >>> prompt = Prompt('Give me two {attribute} numbers', LuckyNumbers)

        ... then we create a fake LLM call that returns random numbers
        >>> fake_ask_ai = lambda _: {'a': 12, 'b': 13}
        >>> bad = BaseStrategy(prompt, ask_ai=fake_ask_ai)
        >>> good = BaseStrategy(prompt, ask_ai=lambda _: {'a': 5, 'b': 5})
        >>> strategy_with_fallback = Fallback(inner=bad, fallback=good)

        ... we can call the fallback strategy, just as other strategies
        >>> strategy_with_fallback(attribute='lucky')
        ({'a': 5, 'b': 5}, True, None, None)

        ... we can also call them with different prompts
        >>> @dataclass
        ... class MoreLuckyNumbers(Schema):
        ...     a: Interval[0, 20]
        ...     b: Interval[0, 20]
        ...     c: Interval[0, 20]

        >>> prompt2 = Prompt('Give me the three random numbers', MoreLuckyNumbers)

        ... to call the strategy with a different prompt:
        Since there are two "inner" strategies, we need to specify which one
        of them gets the new prompt. For this, we use "__" syntax:
        `fallback__<>` are the parameters for the fallback strategy.
        For example:
        >>> strategy_with_fallback(fallback__prompt=prompt2, attribute='lucky')
        ({'a': 5, 'b': 5}, False, '"c" is missing.', '"c" must be present.')

        ... or inner__<> for the inner strategy:
        >>> strategy_with_fallback(inner__prompt=prompt2, attribute='lucky')
        ({'a': 5, 'b': 5}, True, None, None)

        ... or both:
        >>> strategy_with_fallback(inner__prompt=prompt2, fallback__prompt=prompt2, attribute='lucky')
        ({'a': 5, 'b': 5}, False, '"c" is missing.', '"c" must be present.')

        ... we can also log this:
        >>> reply = strategy_with_fallback(
        ...     inner__prompt=prompt2,
        ...     fallback__prompt=prompt2,
        ...     attribute='lucky',
        ...     log=print)
        ===================LOG===================
        [BaseStrategy]ok=False Δt=0.51s
        -----------------------------------------
        [PROMPT]
        Give me the three random numbers
        Fill in **valid JSON** for the fields below.
        <BLANKLINE>
        Rules
        - a must be a number between 0 and 20
        - b must be a number between 0 and 20
        - c must be a number between 0 and 20
        <BLANKLINE>
        Example:
        {
          "a": 10,
          "b": 10,
          "c": 10
        }
        <BLANKLINE>
        Return **only** the JSON object — no code-fences, no comments.
        -----------------------------------------
        [REPLY]
        {
          "a": 12,
          "b": 13
        }
        -----------------------------------------
          ✖ err: "c" is missing.
          ✖ exp: "c" must be present.
        ===================LOG===================
        [BaseStrategy]ok=False Δt=0.51s
        -----------------------------------------
        [PROMPT]
        Give me the three random numbers
        Fill in **valid JSON** for the fields below.
        <BLANKLINE>
        Rules
        - a must be a number between 0 and 20
        - b must be a number between 0 and 20
        - c must be a number between 0 and 20
        <BLANKLINE>
        Example:
        {
          "a": 10,
          "b": 10,
          "c": 10
        }
        <BLANKLINE>
        Return **only** the JSON object — no code-fences, no comments.
        -----------------------------------------
        [REPLY]
        {
          "a": 5,
          "b": 5
        }
        -----------------------------------------
          ✖ err: "c" is missing.
          ✖ exp: "c" must be present.
        ===================LOG===================
        [Fallback]ok=False Δt=1.02s
        -----------------------------------------
        [PROMPT]
        Give me two lucky numbers
        Fill in **valid JSON** for the fields below.
        <BLANKLINE>
        Rules
        - a must be a number between 0 and 10
        - b must be a number between 0 and 10
        <BLANKLINE>
        Example:
        {
          "a": 5,
          "b": 5
        }
        <BLANKLINE>
        Return **only** the JSON object — no code-fences, no comments.
        -----------------------------------------
        [REPLY]
        {
          "a": 5,
          "b": 5
        }
        -----------------------------------------
          ✖ err: "c" is missing.
          ✖ exp: "c" must be present.




    """
    inner: Strategy
    fallback: Strategy
    prompt: Prompt = field(init=False)

    def __post_init__(self):
        self.prompt = self.inner.prompt  # expose primary prompt

    @kwroute
    @override_self
    def __call__(self, inner=None, fallback=None, **tmpl):
        reply, ok, err, exp = self.inner(**tmpl)
        if ok:
            return reply, ok, err, exp
        return self.fallback(**tmpl)


# flexible_strategy.py  (CONTINUATION)  ────────────────────────────────
# ---------------------------------------------------------------------
# Auto-repair wrapper
# ---------------------------------------------------------------------
@dataclass
class AutoRepair(Strategy):
    """
    Wrap a base strategy and keep asking the LLM to repair its JSON until
    the reply validates or the recursion depth is exhausted.

    Example:
        We start by defining a schema and a prompt as before ...
        >>> from dataclasses import dataclass
        >>> from mate_strategy.schema import Schema
        >>> from mate_strategy.rules.predefined import Interval
        >>>
        >>> @dataclass
        ... class RandomNumbers(Schema):
        ...     x: Interval[0, 10]
        >>>
        >>> import random
        >>> fake_llm = lambda _: {'x': random.randint(10, 50)}
        >>>
        >>> from mate_strategy.prompt import Prompt
        >>> from mate_strategy.strategy import BaseStrategy

        ... we can use an AutoRepair strategy to
        >>> prompt = Prompt("Give me a random number", RandomNumbers)
        >>> base   = BaseStrategy(prompt, ask_ai=fake_llm)
        >>> rand   = AutoRepair(base, depth=0)

        ...
        >>> random.seed(42)
        >>> rand()
        ({'x': 50}, False, '"x" is invalid.', '"x" must be a number between 0 and 10')

        >>> random.seed(42)
        >>> rand(depth=1)  # doctest: +ELLIPSIS
        ({'x': 50}, False, '"x" is invalid.', '"x" must be a number between 0 and 10')

        >>> random.seed(42)
        >>> rand(depth=20)  # doctest: +ELLIPSIS
        ({'x': 10}, True, None, None)

        ... again, we can adjust the propmt on the fly
        >>> @dataclass
        ... class BroaderInterval(Schema):
        ...     x: Interval[0, 50]


        >>> prompt_50 = Prompt('Give me a random numbers', BroaderInterval)
        >>> rand(prompt=prompt_50)
        ({'x': 20}, True, None, None)

        ... we can also use more elaborate strategies:
        >>> good_ai = lambda _: {'x': 5}
        >>> bad_ai = lambda _: {'x': 55}
        >>> ai_factory = lambda seq: (lambda _, it=iter(seq): next(it))

        >>> god = BaseStrategy(prompt_50, ask_ai=good_ai)
        >>> bad = BaseStrategy(prompt_50, ask_ai=bad_ai)

        >>> ar_wrap = AutoRepair(Fallback(inner=bad, fallback=god), depth=2)
        >>> ar_wrap()
        ({'x': 5}, True, None, None)

        >>> repair_ai = ai_factory([{"x": 60}, {"x": 70}, {"x": 4}])
        >>> bad_repair = BaseStrategy(prompt_50, ask_ai=repair_ai)
        >>> ar_inner = AutoRepair(bad_repair, depth=3)
        >>> combo = Fallback(inner=ar_inner, fallback=bad)
        >>> combo()
        ({'x': 4}, True, None, None)

    """
    inner: Strategy
    depth: int = 1
    mode: str = "sub"  # "sub" → only FIX text, "full" → prepend original
    repair_retries: int = 2
    repair_backoff: float = 0.0  # seconds

    prompt: Prompt = field(init=False)  # cached original prompt

    # ───────────────────────── init ─────────────────────────
    def __post_init__(self) -> None:
        self.prompt = self.inner.prompt

    # ───────────────────────── public API ─────────────────────────
    @kwroute
    @override_self
    def __call__(self,
                 *,  # keyword-only
                 inner: Strategy | None = None,
                 depth: int | None = None,
                 mode: str | None = None,
                 prompt: Prompt | None = None,
                 **tmpl):
        inner = inner or self.inner
        depth = self.depth if depth is None else depth
        mode = self.mode if mode is None else mode

        # override prompt (if given) by wrapping a temporary BaseStrategy
        if prompt is not None:
            inner = BaseStrategy(prompt,
                                 ask_ai=inner.ask_ai,
                                 retries=inner.retries,
                                 backoff=inner.backoff)

        runner_prompt = inner.prompt  # prompt actually sent to LLM
        schema = runner_prompt.schema

        reply, ok, err, exp = inner(**tmpl)
        if ok or depth == 0:
            return reply, ok, err, exp

        fixed = copy.deepcopy(reply)
        if self._repair(fixed, tmpl,
                        depth=depth,
                        mode=mode,
                        prompt=runner_prompt,
                        schema=schema):
            return fixed, True, None, None
        return reply, False, err, exp

    # ─────────────────────── internal logic ───────────────────────
    def _repair(
            self,
            repl: dict,
            tmpl: dict,
            *,
            depth: int,
            mode: str,
            prompt: Prompt,
            schema,
    ) -> bool:
        """Mutate `repl` in-place; return True ⇢ validation succeeded."""
        if depth == 0:
            return False

        ok, err, expl = prompt.validate(repl)
        if ok:
            return True

        # -------- build repair prompt --------
        fix_txt = schema.repair_prompt(repl)

        if mode == "sub":
            repair_txt = fix_txt
        else:
            repair_txt = prompt.render(**tmpl) + "\n---\nFIX:\n" + fix_txt

        # escape braces so .format() in Prompt.render won’t choke
        repair_txt = repair_txt.replace("{", "{{").replace("}", "}}")

        repair_prompt = Prompt(repair_txt, schema)
        repair_strategy = BaseStrategy(
            repair_prompt,
            ask_ai=self.inner.ask_ai,
            retries=self.repair_retries,
            backoff=self.repair_backoff,
        )

        new_reply, ok2, _, _ = repair_strategy()
        if ok2:
            repl.clear()
            repl.update(new_reply)
            return True

        if self.repair_backoff:
            time.sleep(self.repair_backoff)

        # recurse ↓
        return self._repair(
            repl, tmpl,
            depth=depth - 1,
            mode=mode,
            prompt=prompt,
            schema=schema,
        )

    # ─────────────────────── misc helpers (unchanged) ───────────────────────
    def _parts(self, path: str):
        return [p for p in path.replace("]", "").replace("[", ".").split(".") if p]

    def _extract(self, data, path):
        cur = data
        for p in self._parts(path):
            cur = cur[int(p)] if p.isdigit() else cur[p]
        return copy.deepcopy(cur)

    def _set(self, data, path, value):
        parts = self._parts(path)
        cur = data
        for p in parts[:-1]:
            cur = cur[int(p)] if p.isdigit() else cur[p]
        last = parts[-1]
        if last.isdigit():
            cur[int(last)] = value
        else:
            cur[last] = value
