from typing import Callable, Optional

import mlflow
import mlflow.entities


def set_project(
    original: Callable,
    name: str,
    mlflowclient: Optional[mlflow.MlflowClient] = None,
) -> None:
    mlflowclient = mlflowclient or mlflow.MlflowClient()
    set_project_method: Callable = original

    set_project_method(name)

    if not mlflowclient.search_experiments(filter_string=f"name='{name}'"):
        mlflowclient.create_experiment(name=name)
