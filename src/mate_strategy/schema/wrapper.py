# mate_strategy/wrapper.py  (core, domain-agnostic)
from abc import abstractmethod
from typing import TypeVar, Generic, Any, Dict

T = TypeVar("T")


class GenericWrapper(Generic[T]):
    Schema: type["Schema"]  # subclasses set this

    def __init__(self, inner: T):
        self._inner = inner

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GenericWrapper[T]":
        cls.Schema.validate_or_raise(d)
        return cls(cls._to_domain_impl(d))  # HOOK #1

    def json(self) -> Dict[str, Any]:
        return self._to_dict_impl(self._inner)  # HOOK #2

    @abstractmethod
    def _to_domain_impl(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _to_dict_impl(self, inner: T) -> Dict[str, Any]:
        pass
