from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

from typing_extensions import ParamSpec

CallableArgsTypes = ParamSpec("CallableArgsTypes")
CallableReturnType = TypeVar("CallableReturnType")


def with_kwargs(
    **extra_kwargs: Any,
) -> Callable[
    [Callable[CallableArgsTypes, CallableReturnType]],
    Callable[CallableArgsTypes, CallableReturnType],
]:
    def decorator(
        function: Callable[CallableArgsTypes, CallableReturnType],
    ) -> Callable[CallableArgsTypes, CallableReturnType]:
        @functools.wraps(function)
        def wrapper(
            *args: CallableArgsTypes.args, **kwargs: CallableArgsTypes.kwargs
        ) -> CallableReturnType:
            kwargs.update(extra_kwargs)
            return function(*args, **kwargs)

        return wrapper

    return decorator
