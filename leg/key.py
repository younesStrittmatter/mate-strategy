from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Union
from src.mate_strategy.json.rules import Rule


# ─────────────────────────────── validators
@dataclass(frozen=True)
class Key:
    name: str
    type_: type | None = None
    rule: Union[Rule, Callable[[Any], bool]] | None = None
    desc: str | None = None

    def __call__(self, value: Any) -> bool:
        if self.type_ and not isinstance(value, self.type_):
            try:
                value = self.type_(value)
            except (ValueError, TypeError):
                return False
        return self.rule(value) if self.rule else True

    def rule_line(self) -> str:
        # e.g. • "id" must be an integer between 1 and 100.
        line = getattr(self.rule, "rule_line", "")
        if line == "":
            if self.type_:
                line = f"must be a {self.type_.__name__}"
            else:
                line = "must be a value"
        return f'• "{self.name}" {line}.'

    def placeholder(self) -> str:
        ph = getattr(self.rule, "desc", "")
        if not ph == "":
            if self.type_:
                ph = f"<{self.type_.__name__} {ph}>"
            else:
                ph = "<value>"
        else:
            if self.type_:
                ph = f"<{self.type_.__name__} …>"
            else:
                ph = "<value>"
        return ph

    def __repr__(self):
        typ = self.type_.__name__ if self.type_ else "Any"
        rule = getattr(self.rule, "desc", "")
        return f'"{self.name}": …  \\\\ {typ}; {rule}'
