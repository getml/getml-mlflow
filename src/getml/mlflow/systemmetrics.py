import threading
from enum import StrEnum
from typing import Dict, Optional

import numpy
import requests

import mlflow


class Metric(StrEnum):
    CPU_USAGE = "engine_cpu_usage_per_virtual_core_in_pct"
    MEMORY_USAGE = "memory_usage_in_pct"


class SystemMetrics:
    HOST: str = "localhost"
    PORT: int = 1709

    def __init__(self, run_id: str, host: str = HOST, port: int = PORT):
        self._run_id = run_id
        self._event: threading.Event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        url: str = f"http://{host}:{port}"
        self._metrics_endpoints: Dict[Metric, str] = {
            Metric.CPU_USAGE: f"{url}/getcpuusage/",
            Metric.MEMORY_USAGE: f"{url}/getmemoryusage/",
        }

    def _run_logging_metrics(self) -> None:
        step: int = 0
        metrics_endpoints: Dict[Metric, str] = self._valid_metrics_endpoints()
        while not self._event.is_set():
            self._log_metrics(step, metrics_endpoints)
            step += 1
            self._event.wait(1)

    def _log_metrics(self, step: int, metrics_endpoints: Dict[Metric, str]):
        metrics: Dict[str, float] = {}
        for metric_name, metric_url in metrics_endpoints.items():
            try:
                response = requests.get(metric_url)
                metrics[metric_name] = numpy.round(response.json()["data"][0][-1], 2)
            except requests.exceptions.RequestException as exception:
                mlflow.log_text(
                    f"Exception on GET({metric_url}): {exception}",
                    "error.log",
                    run_id=self._run_id,
                )
                continue
        if metrics:
            mlflow.log_metrics(
                run_id=self._run_id,
                metrics=metrics,
                step=step,
            )

    def _valid_metrics_endpoints(self) -> Dict[Metric, str]:
        valid_metrics_endpoints: Dict[Metric, str] = {}
        for name, endpoint in self._metrics_endpoints.items():
            try:
                response: requests.Response = requests.get(endpoint)
                if response.ok:
                    valid_metrics_endpoints[name] = endpoint
            except requests.exceptions.RequestException as exception:
                print("Engine metrics are available in the Enterprise edition.")
                mlflow.log_text(
                    f"Exception on GET({endpoint}): {exception}",
                    "error.log",
                    run_id=self._run_id,
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
