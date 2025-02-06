from typing import Callable

from getml_mlflow.loggingconfiguration import LoggingConfiguration


def set_project(
    original: Callable,
    name: str,
    *,
    logging_configuration: LoggingConfiguration = LoggingConfiguration(),
) -> None:
    mlflowclient = logging_configuration.mlflowclient
    set_project_method: Callable = original

    set_project_method(name)

    if not mlflowclient.search_experiments(filter_string=f"name='{name}'"):
        mlflowclient.create_experiment(name=name)
