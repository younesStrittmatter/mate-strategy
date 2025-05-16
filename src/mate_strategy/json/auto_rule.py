# mate_strategy/self_heal/autorule.py
from __future__ import annotations

import json

from typing import Any
from dataclasses import dataclass

from src.mate_strategy.json.rules import Rule  # the protocol you defined
from src.mate_strategy.json.parseable import Parseable



@dataclass(frozen=True)
class AutoRule(Rule):
    """
    Turn an arbitrary Parseable class into a Rule object that

    • Generates nice rule-lines / placeholders for prompt building
    • Validates:  cls.from_dict()   (True ↔ parses OK)
    • Can emit a *repair prompt* for self-healing
    """
    cls: type[Parseable]

    # ---------- Rule interface ---------------------------------
    @property
    def desc(self) -> str:
        return f"{self.cls.__name__} object"

    @property
    def rule_line(self) -> str:
        return f"must be a {self.cls.__name__} object {self._brief()}"

    def __call__(self, x: Any) -> bool:
        if not isinstance(x, dict):
            return False
        try:
            self.cls.from_dict(x)
            return True
        except Exception:  # parsing failed
            return False

    # ---------- helper for Response.prompt() -------------------
    def placeholder(self) -> str:
        return f"{{ …{self.cls.__name__}… }}"

    # ---------- self-repair helpers ----------------------------
    def repair_prompt(self, bad_json: dict) -> str:
        """Return a *small* prompt asking to fix only this sub-object."""
        sub_schema_lines = _brief_schema(self.cls)
        return (
            "The JSON below is invalid – please fix **only** this part.\n"
            f"Rules for a valid {self.cls.__name__}:\n{sub_schema_lines}\n\n"
            f"Current value (invalid):\n```json\n{json.dumps(bad_json, indent=2)}\n```\n"
            "Reply with **only** the corrected JSON object."
        )

    # ---------- internals --------------------------------------
    def _brief(self) -> str:
        # show field names only
        names = ", ".join(k.name for k in self.cls.__schema__)
        return f"{{ {names} }}"


def _brief_schema(cls: type[Parseable]) -> str:
    return "\n".join("• " + k.rule_line() for k in cls.__schema__)
