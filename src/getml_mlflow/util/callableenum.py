from __future__ import annotations

from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, Type, TypeVar, cast

CallableType = TypeVar("CallableType", bound=Callable[..., Any])


class CallableEnumFactory(Generic[CallableType]):
    @classmethod
    def build(cls, name: str, entries: Dict[str, CallableType]) -> Type[Enum]:
        enum = Enum(name, {key: (auto(), value) for key, value in entries.items()})
        setattr(enum, "value", property(lambda self: self._value_[1]))
        return cast(Type[Enum], enum)
