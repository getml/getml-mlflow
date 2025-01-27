import json
from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Literal, Optional

import getml
import mlflow

PARAMETER_NAMES = (
    "preprocessors",
    "feature_learners",
    "feature_selectors",
    "predictors",
    "loss_function",
    "include_categorical",
    "share_selected_features",
)


def log_parameters(pipeline: getml.Pipeline, run_id: str) -> None:
    parameters = {}

    for parameter_name in PARAMETER_NAMES:
        parameter = getattr(pipeline, parameter_name)
        if is_dataclass(type(parameter)):
            parameters.update(_serialize_dataclass(parameter_name, parameter))
        elif isinstance(parameter, list):
            for id, item in enumerate(parameter):
                parameters.update(_serialize_dataclass(f"{parameter_name}.{id}", item))
        elif isinstance(parameter, (str, int, float, bool, Literal)):
            parameters[parameter_name] = parameter
        else:
            parameters[parameter_name] = str(parameter)

    mlflow.log_params(params=parameters, run_id=run_id)


def _serialize_dataclass(name: str, parameter: Any) -> Dict[str, Any]:
    parameters: Dict[str, Any] = {}

    current_name: str = parameter.__class__.__name__
    for field in fields(parameter):
        full_field_name: str = f"{name}.{current_name}.{field.name}"
        field_value: Any = getattr(parameter, field.name)
        if is_dataclass(field.type):
            parameters.update(_serialize_dataclass(full_field_name, field_value))
        else:
            parameters[full_field_name] = _serialize_field_value(field_value)

    return parameters


@staticmethod
def _serialize_field_value(field_value: Any) -> str:
    if isinstance(field_value, (frozenset, set)):
        return json.dumps(sorted(field_value))
    if not isinstance(field_value, str):
        return json.dumps(field_value)
    return field_value


@staticmethod
def log_tags(pipeline: getml.Pipeline) -> None:
    tags: Dict[str, str] = {"id": pipeline.id}
    for tag in map(str, pipeline.tags):
        if ":" in tag:
            key, value = tag.split(":")
            tags[key.strip()] = value.strip()
        else:
            tags[tag] = tag
    mlflow.set_tags(tags=tags)


def log_metrics(pipeline: getml.Pipeline, run_id: str) -> None:
    metrics = {}

    scores = pipeline.scores

    if pipeline.is_classification:
        metrics.update(_serialize_metric("auc", scores.auc, 2))
        metrics.update(_serialize_metric("accuracy", scores.accuracy, 2))
        metrics.update(_serialize_metric("cross_entropy", scores.cross_entropy, 4))

    if pipeline.is_regression:
        metrics.update(_serialize_metric("mae", scores.mae))
        metrics.update(_serialize_metric("rmse", scores.rmse))
        metrics.update(_serialize_metric("rsquared", scores.rsquared, 2))

    # TODO: Add feature importance and correlation
    # for feature in pipeline.features:
    #     metrics[f"{feature.name}.importance"] = json.dumps(feature.importance)
    #     metrics[f"{feature.name}.correlation"] = json.dumps(feature.correlation)

    # if len(pipeline.targets) == 1:
    #     metrics["targets"] = getml_pipeline.targets[0]
    # else:
    #     for i, t in enumerate(getml_pipeline.targets):
    #         metrics[f"targets.{i}"] = t

    mlflow.log_metrics(metrics=metrics, run_id=run_id)


def _serialize_metric(
    name: str, values: float | List[float], ndigits: Optional[int] = None
) -> Dict[str, float]:
    if isinstance(values, list):
        return {
            f"name.{id}": maybe_round(value, ndigits) for id, value in enumerate(values)
        }
    return {name: maybe_round(values, ndigits)}


def maybe_round(value: float, ndigits: Optional[int]) -> float:
    return value if ndigits is None else round(value, ndigits)


def set_id_tag(pipeline: getml.Pipeline, run_id: str) -> None:
    if (active_run := mlflow.active_run()) and active_run.info.run_id == run_id:
        mlflow.set_tag(key="id", value=pipeline.id)
    else:
        with mlflow.start_run(
            run_id=run_id,
            run_name="Pipeline",
            nested=False,
        ):
            mlflow.set_tag(key="id", value=pipeline.id)
