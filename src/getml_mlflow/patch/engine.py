from typing import Callable

import mlflow
import mlflow.entities


def set_project(original: Callable, name: str) -> None:
    set_project_method: Callable = original
    set_project_method(name)
    if not mlflow.search_experiments(filter_string=f"name='{name}'"):
        mlflow.create_experiment(name=name)
    mlflow.set_experiment(experiment_name=name)
