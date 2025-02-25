from __future__ import annotations

import logging
from datetime import datetime, timezone
from logging import Logger
from types import TracebackType
from typing import Optional, Type

from mlflow import MlflowClient

logger: Logger = logging.getLogger(__name__)


def log_exit_exception(
    mlflow_client: MlflowClient,
    run_id: Optional[str],
    exc_type: Type[BaseException],
    exc_val: BaseException,
    exc_tb: Optional[TracebackType],
) -> None:
    if run_id is not None:
        mlflow_client.log_text(
            run_id=run_id,
            text=f"Exception: {exc_type}: {exc_val}",
            artifact_file=f"error/{datetime.now(timezone.utc).isoformat()}.log",
        )
    logger.error(
        f"Exception: {exc_type}: {exc_val}",
        exc_info=(exc_type, exc_val, exc_tb),
    )


def log_request_exception(
    mlflow_client: MlflowClient,
    run_id: str,
    exception: BaseException,
    context: str,
) -> None:
    mlflow_client.log_text(
        run_id=run_id,
        text=f"Exception: {context}: {exception}",
        artifact_file=f"error/{datetime.now(timezone.utc).isoformat()}.log",
    )
