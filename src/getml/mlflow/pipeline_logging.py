import json
from dataclasses import fields, is_dataclass
from typing import Any, Callable, Dict, List, Literal, Optional

import getml
import mlflow


class PipelineLogging:
    PIPELINE_PARAMETER_NAMES = (
        "preprocessors",
        "feature_learners",
        "feature_selectors",
        "predictors",
        "loss_function",
        "include_categorical",
        "share_selected_features",
    )

    @classmethod
    def log_parameters(cls, pipeline: getml.Pipeline, run_id: str) -> None:
        parameters = {}

        for parameter_name in cls.PIPELINE_PARAMETER_NAMES:
            parameter = getattr(pipeline, parameter_name)
            if is_dataclass(type(parameter)):
                parameters.update(cls._serialize_dataclass(parameter_name, parameter))
            elif isinstance(parameter, list):
                for id, item in enumerate(parameter):
                    parameters.update(
                        cls._serialize_dataclass(f"{parameter_name}.{id}", item)
                    )
            elif isinstance(parameter, (str, int, float, bool, Literal)):
                parameters[parameter_name] = parameter
            else:
                parameters[parameter_name] = str(parameter)

        mlflow.log_params(params=parameters, run_id=run_id)

    @classmethod
    def _serialize_dataclass(cls, name: str, parameter: Any) -> Dict[str, Any]:
        parameters: Dict[str, Any] = {}

        current_name: str = parameter.__class__.__name__
        for field in fields(parameter):
            full_field_name: str = f"{name}.{current_name}.{field.name}"
            field_value: Any = getattr(parameter, field.name)
            if is_dataclass(field.type):
                parameters.update(
                    cls._serialize_dataclass(full_field_name, field_value)
                )
            else:
                parameters[full_field_name] = cls._serialize_field_value(field_value)

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
        tags: Dict[str, str] = {}
        index: int = 0
        for tag in map(str, pipeline.tags):
            if ":" in tag:
                key, value = tag.split(":")
                tags[key.strip()] = value.strip()
            else:
                tags[str(index)] = tag
                index = index + 1
        mlflow.set_tags(tags=tags)

    @classmethod
    def log_metrics(cls, pipeline: getml.Pipeline, run_id: str) -> None:
        metrics = {}

        scores = pipeline.scores

        if pipeline.is_classification:
            metrics.update(cls._serialize_metric("auc", scores.auc, 2))
            metrics.update(cls._serialize_metric("accuracy", scores.accuracy, 2))
            metrics.update(
                cls._serialize_metric("cross_entropy", scores.cross_entropy, 4)
            )

        if pipeline.is_regression:
            metrics.update(cls._serialize_metric("mae", scores.mae))
            metrics.update(cls._serialize_metric("rmse", scores.rmse))
            metrics.update(cls._serialize_metric("rsquared", scores.rsquared, 2))

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

    @staticmethod
    def _serialize_metric(
        name: str, values: float | List[float], ndigits: Optional[int] = None
    ) -> Dict[str, float]:
        maybe_round: Callable[[float], float] = (
            lambda x: x if ndigits is None else round(x, ndigits)
        )
        if isinstance(values, list):
            return {f"name.{id}": maybe_round(value) for id, value in enumerate(values)}
        return {name: maybe_round(values)}
