import logging
import threading
from enum import StrEnum
from typing import Dict, List, Optional

import mlflow
import mlflow.entities
import mlflow.utils.time
import numpy
import requests


class Metric(StrEnum):
    CPU_USAGE = "engine_cpu_usage_per_virtual_core_in_pct"
    MEMORY_USAGE = "memory_usage_in_pct"


class SystemMetrics:
    HOST: str = "localhost"
    PORT: int = 1709

    def __init__(
        self,
        run_id: str,
        host: str = HOST,
        port: int = PORT,
        mlflowclient: Optional[mlflow.MlflowClient] = None,
    ):
        self._run_id: str = run_id
        self._event: threading.Event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        url: str = f"http://{host}:{port}"
        self._metrics_endpoints: Dict[Metric, str] = {
            Metric.CPU_USAGE: f"{url}/getcpuusage/",
            Metric.MEMORY_USAGE: f"{url}/getmemoryusage/",
        }
        self._mlflowclient = mlflowclient or mlflow.MlflowClient()

    def _run_logging_metrics(self) -> None:
        step: int = 0
        metrics_endpoints: Dict[Metric, str] = self._valid_metrics_endpoints()
        while not self._event.is_set():
            self._log_metrics(step, metrics_endpoints)
            step += 1
            self._event.wait(1)

    def _log_metrics(self, step: int, metrics_endpoints: Dict[Metric, str]):
        metrics: List[mlflow.entities.Metric] = []
        timestamp: int = mlflow.utils.time.get_current_time_millis()
        for metric_name, metric_url in metrics_endpoints.items():
            try:
                response: requests.Response = requests.get(metric_url)
                metrics.append(
                    mlflow.entities.Metric(
                        key=metric_name,
                        value=numpy.round(response.json()["data"][0][-1], 2),
                        timestamp=timestamp,
                        step=step,
                    )
                )
            except requests.exceptions.RequestException as exception:
                self._mlflowclient.log_text(
                    run_id=self._run_id,
                    text=f"Exception on GET({metric_url}): {exception}",
                    artifact_file="error.log",
                )
                continue
        if metrics:
            self._mlflowclient.log_batch(
                run_id=self._run_id,
                metrics=metrics,
            )

    def _valid_metrics_endpoints(self) -> Dict[Metric, str]:
        valid_metrics_endpoints: Dict[Metric, str] = {}
        for name, endpoint in self._metrics_endpoints.items():
            try:
                response: requests.Response = requests.get(endpoint)
                if response.ok:
                    valid_metrics_endpoints[name] = endpoint
            except requests.exceptions.RequestException as exception:
                logging.getLogger("getML").warn(
                    "Engine metrics are available in the Enterprise edition. "
                    "Visit https://getml.com/latest/enterprise/ for more information"
                )
                self._mlflowclient.log_text(
                    run_id=self._run_id,
                    text=f"Exception on GET({endpoint}): {exception}",
                    artifact_file="error.log",
                )
                continue
        return valid_metrics_endpoints

    def __enter__(self):
        self._thread = threading.Thread(target=self._run_logging_metrics)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._event.set()
        if self._thread is not None:
            self._thread.join()
        self._thread = None
