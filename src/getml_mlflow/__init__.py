from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List

from getml_mlflow.autologging import autolog

__all__: List[str] = [
    "autolog",
]
