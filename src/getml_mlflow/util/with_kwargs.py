from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable

import functools


def with_kwargs(
    **extra_kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            kwargs.update(extra_kwargs)
            return func(*args, **kwargs)

        return wrapper

    return decorator
