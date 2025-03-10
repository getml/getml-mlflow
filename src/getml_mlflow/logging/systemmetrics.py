from __future__ import annotations

import logging
from collections import namedtuple
from logging import Logger
from threading import Event, Thread
from types import TracebackType
from typing import Callable, List, Optional, Type

import mlflow.utils.time
import numpy
import requests
from getml.constants import ENTERPRISE_DOCS_URL
from mlflow import MlflowClient
from mlflow.entities import Metric
from requests import Response
from requests.exceptions import RequestException

from getml_mlflow.constants import DEFAULT_GETML_HOST, DEFAULT_GETML_PORT
from getml_mlflow.logging.logger import log_exit_exception, log_request_exception
from getml_mlflow.util.callableenum import CallableEnumFactory

logger: Logger = logging.getLogger(__name__)


EngineMetric = CallableEnumFactory[Callable[[str], str]].build(
    "EngineMetric",
    dict(
        ENGINE_CPU_USAGE_PER_VIRTUAL_CORE_IN_PCT=lambda url: f"{url}/getcpuusage/",
        MEMORY_USAGE_IN_PCT=lambda url: f"{url}/getmemoryusage/",
    ),
)
ValidEngineMetric = namedtuple("ValidEngineMetric", ["name", "url"])


class SystemMetricsLogger:
    HOST: str = DEFAULT_GETML_HOST
    PORT: int = DEFAULT_GETML_PORT

    def __init__(
        self,
        mlflow_client: MlflowClient,
        run_id: str,
        host: str = HOST,
        port: int = PORT,
        *,
        log_system_metrics: bool = True,
    ) -> None:
        self._run_id: str = run_id
        self._event: Event = Event()
        self._thread: Optional[Thread] = None

        self._url: str = f"http://{host}:{port}"
        self._mlflow_client: MlflowClient = mlflow_client
        self._log_system_metrics: bool = log_system_metrics

    def _run_logging_metrics(self) -> None:
        step: int = 0
        valid_engine_metrics: List[ValidEngineMetric] = self._valid_metrics()
        while not self._event.is_set():
            self._log_metrics(step, valid_engine_metrics)
            step += 1
            self._event.wait(1)

    def _log_metrics(
        self, step: int, valid_engine_metrics: List[ValidEngineMetric]
    ) -> None:
        metrics: List[Metric] = []
        timestamp: int = mlflow.utils.time.get_current_time_millis()
        for engine_metric_name, engine_metric_url in valid_engine_metrics:
            try:
                response: Response = requests.get(engine_metric_url)
                metrics.append(
                    Metric(
                        key=engine_metric_name,
                        value=numpy.round(response.json()["data"][0][-1], 2),
                        timestamp=timestamp,
                        step=step,
                    )
                )
            except RequestException as exception:
                log_request_exception(
                    self._mlflow_client,
                    self._run_id,
                    exception,
                    f"GET({engine_metric_url})",
                )
                continue
        if metrics:
            self._mlflow_client.log_batch(
                run_id=self._run_id,
                metrics=metrics,
            )

    def _valid_metrics(self) -> List[ValidEngineMetric]:
        valid_engine_metrics: List[ValidEngineMetric] = []
        for engine_metric in EngineMetric:
            engine_metric_endpoint: str = engine_metric.value(self._url)
            try:
                response: Response = requests.get(engine_metric_endpoint)
                if response.ok:
                    valid_engine_metrics.append(
                        ValidEngineMetric(
                            engine_metric.name.lower(), engine_metric_endpoint
                        )
                    )
                else:
                    response.raise_for_status()
            except requests.exceptions.RequestException as exception:
                log_request_exception(
                    self._mlflow_client,
                    self._run_id,
                    exception,
                    f"GET({engine_metric_endpoint})",
                )
                logger.warning(
                    f"Engine metrics ({engine_metric_endpoint}) are available in the Enterprise edition. "
                    f"Visit {ENTERPRISE_DOCS_URL} for more information"
                )
                continue
        return valid_engine_metrics

    def __enter__(self) -> SystemMetricsLogger:
        if not self._log_system_metrics:
            return self

        self._thread = Thread(target=self._run_logging_metrics)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
    ) -> None:
        if exc_type is not None and exc_value is not None:
            log_exit_exception(
                self._mlflow_client, self._run_id, exc_type, exc_value, exc_traceback
            )

        if not self._log_system_metrics:
            return

        self._event.set()
        if self._thread is not None:
            self._thread.join()
        self._thread = None
