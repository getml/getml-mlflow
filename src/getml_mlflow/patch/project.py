from __future__ import annotations

from typing import Callable

from mlflow import MlflowClient

from getml_mlflow.loggingconfiguration import LoggingConfiguration


def switch(
    original: Callable,
    name: str,
    *,
    logging_configuration: LoggingConfiguration = LoggingConfiguration(),
) -> None:
    mlflow_client: MlflowClient = logging_configuration.mlflow_client
    switch_method: Callable = original

    switch_method(name)

    if not mlflow_client.search_experiments(filter_string=f"name='{name}'"):
        mlflow_client.create_experiment(name=name)
