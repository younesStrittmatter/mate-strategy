# mate_strategy/wrapper.py
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Dict, Any
from mate_strategy.schema import Schema  # the base you just updated

T = TypeVar("T")


class GenericWrapper(Generic[T], ABC):
    """Domain-agnostic helper for JSON ⇆ domain round-trips."""
    Schema: type[Schema]  # subclasses *must* set this

    def __init__(self, inner: T):
        self._inner = inner

    # ---------- factory ----------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenericWrapper[T]":
        cls.Schema.validate_or_raise(data)  # uses your Schema core
        return cls(cls._to_domain_impl(data))  # HOOK 1

    # ---------- serialise ----------
    def json(self) -> Dict[str, Any]:
        return self._to_dict_impl(self._inner)  # HOOK 2

    # ---------- hooks (static, abstract) ----------
    @staticmethod
    @abstractmethod
    def _to_domain_impl(data: Dict[str, Any]) -> T: ...

    @staticmethod
    @abstractmethod
    def _to_dict_impl(obj: T) -> Dict[str, Any]: ...
