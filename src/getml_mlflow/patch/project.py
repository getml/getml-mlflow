from typing import Callable

from mlflow import MlflowClient

from getml_mlflow.loggingconfiguration import LoggingConfiguration


def switch(
    original: Callable,
    name: str,
    *,
    logging_configuration: LoggingConfiguration = LoggingConfiguration(),
) -> None:
    mlflowclient: MlflowClient = logging_configuration.mlflowclient
    switch_method: Callable = original

    switch_method(name)

    if not mlflowclient.search_experiments(filter_string=f"name='{name}'"):
        mlflowclient.create_experiment(name=name)
