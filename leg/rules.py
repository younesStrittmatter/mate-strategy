from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Protocol


class Rule(Protocol):
    def __call__(self, value: Any) -> bool: ...

    @property
    @abstractmethod
    def desc(self) -> str: ...

    @property
    @abstractmethod
    def rule_line(self) -> str: ...


@dataclass(frozen=True)
class Interval(Rule):
    lo: float
    hi: float

    def __call__(self, x: Any) -> bool:
        return isinstance(x, (int, float)) and self.lo <= x <= self.hi

    @property
    def desc(self) -> str:
        return f"{self.lo}-{self.hi}"

    @property
    def rule_line(self) -> str:
        return f"must be an integer between {self.lo} and {self.hi}"


@dataclass(frozen=True)
class OneOf(Rule):
    values: Iterable
    max_examples: int = 4

    def __call__(self, x: Any) -> bool:
        return x in self.values

    @property
    def desc(self) -> str:
        values = set(self.values)
        sorted_values = sorted(values)
        txt = ", ".join(sorted_values[:self.max_examples])
        if len(values) > self.max_examples:
            txt += "…"
        return txt

    @property
    def rule_line(self) -> str:
        if len(set(self.values)) <= self.max_examples:
            return f"must be **one of** these examples: {self.desc}"
        return f"must be exactly **one of** {self.desc}"


