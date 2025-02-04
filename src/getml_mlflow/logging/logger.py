import logging
from datetime import datetime
from types import TracebackType
from typing import Optional, Type

import mlflow


def set_up():
    logger = logging.getLogger("getML")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="{asctime} {levelname} getML: {message}",
        style="{",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def log_exit_exception(
    mlflowclient: mlflow.MlflowClient,
    run_id: Optional[str],
    exc_type: Type[BaseException],
    exc_val: BaseException,
    exc_tb: Optional[TracebackType],
):
    if run_id is not None:
        mlflowclient.log_text(
            run_id=run_id,
            text=f"Exception: {exc_type}: {exc_val}",
            artifact_file=f"error/{datetime.utcnow().isoformat()}.log",
        )
    logging.getLogger("getML").error(
        f"Exception: {exc_type}: {exc_val}",
        exc_info=(exc_type, exc_val, exc_tb),
    )


def log_request_exception(
    mlflowclient: mlflow.MlflowClient,
    run_id: str,
    exception: BaseException,
    context: str,
):
    mlflowclient.log_text(
        run_id=run_id,
        text=f"Exception: {context}: {exception}",
        artifact_file=f"error/{datetime.utcnow().isoformat()}.log",
    )
