from typing import TypeVar, Generic, Any

T = TypeVar("T")


class GenericWrapper(Generic[T]):
    """Domain-agnostic base for JSON⇆object round-trips."""
    Schema: type["Schema"]

    def __init__(self, inner: T):
        self._inner = inner

    # ---- serialise ----
    def json(self) -> dict[str, Any]:
        return self._to_dict_impl(self._inner)

    # ---- de-serialise ----
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenericWrapper[T]":
        cls.Schema.validate_or_raise(data)
        return cls(cls._to_domain_impl(data))

    # ---- hooks to override in adapter layer ----
    @staticmethod
    def _to_domain_impl(data): ...

    @staticmethod
    def _to_dict_impl(obj): ...
