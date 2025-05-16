from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
from src.mate_strategy.json.key import Key


@dataclass(frozen=True)
class Response:
    """
    Bundle of *Keys* that can
    - **validate** a JSON dict
    - **emit** a ready-to-copy LLM prompt snippet

    :param
    keys: Iterable[Key]
        A sequence of :class:`mate_schema.template.json.Key` objects.
        Each *Key* names a JSON field and (optionally) constrains it via:
        1. **Built-in Rule objects**
           ``Interval(lo, hi)``, ``OneOf(iterable)``, …
        2. **Inline lambda / function** (quick but no extra metadata)
           ``lambda x: isinstance(x, bool)``
        3. **Ad-hoc helper that returns a lambda *with* metadata**
    system_prompt: str(optional)
        A string that will be prepended to the prompt.

    Examples:
        >>> from mate_strategy.json.rules import Interval, OneOf
        >>> from mate_strategy.json.key  import Key
        >>> from mate_strategy.json.response import Response
        >>> response = Response([
        ...     Key("id",   int,  Interval(1, 100)),
        ...     Key("name", str,  OneOf(["Janet", "Alice", "Andrew", "Caroline"])),
        ...     Key("flag", bool, lambda x: isinstance(x, bool)),           # simple lambda
        ... ])
        >>> print(response.prompt())         # doctest: +NORMALIZE_WHITESPACE
        Fill in **valid JSON** for the fields below.
        Rules
        • "id" must be an integer between 1 and 100.
        • "name" must be **one of** these examples: Alice, Andrew, Caroline, Janet.
        • "flag" must be a bool.
        <BLANKLINE>
        Return **only** the JSON object — no code-fences, no comments.
        <BLANKLINE>
        Example of correct shape (values are placeholders):
        { "id": <int 1-100>, "name": <str Alice, Andrew, Caroline, Janet>, "flag": <bool …> }
        >>> ok = {"id": 42, "name": "Alice", "flag": True}
        >>> response(ok)
        True


       """
    keys: Iterable[Key]
    system_prompt: str = ""

    def __init__(self, keys: Iterable[Key], system_prompt: str = ""):
        object.__setattr__(self, "keys", tuple(keys))
        object.__setattr__(self, "system_prompt", system_prompt)

    # ---------- validation ----------
    def __call__(self, data: dict) -> bool:
        for k in self.keys:
            if k.name not in data:
                return False
            if not k(data[k.name]):
                return False
        return all(k.name in data and k(data[k.name]) for k in self.keys)

    # ---------- pretty template for the LLM ----------
    def prompt(self) -> str:
        rules = "\n".join(k.rule_line() for k in self.keys)
        # example line
        example_items = ", ".join(
            f'"{k.name}": {k.placeholder()}' for k in self.keys
        )
        example = f"{{ {example_items} }}"
        _s = ""
        if bool(self.system_prompt):
            _s = f"{self.system_prompt}\n"
        return (
            f"{_s}"
            "Fill in **valid JSON** for the fields below.\n"
            "Rules\n"
            f"{rules}\n\n"
            "Return **only** the JSON object — no code-fences, no comments.\n\n"
            "Example of correct shape (values are placeholders):\n"
            f"{example}"
        )

    def __repr__(self):
        inner = ",\n  ".join(repr(k) for k in self.keys)
        return "{\n  " + inner + "\n}"
