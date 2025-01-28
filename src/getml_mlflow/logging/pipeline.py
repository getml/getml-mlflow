import json
from dataclasses import fields, is_dataclass
from typing import Any, List, Literal, Optional

import getml
import mlflow
import mlflow.entities
from mlflow.entities import Metric, Param, RunTag
from mlflow.utils.time import get_current_time_millis

PARAMETER_NAMES = (
    "preprocessors",
    "feature_learners",
    "feature_selectors",
    "predictors",
    "loss_function",
    "include_categorical",
    "share_selected_features",
)


class PipelineLogger:
    @classmethod
    def of_autolog(cls, pipeline: getml.Pipeline, mlflowclient: mlflow.MlflowClient):
        return cls(pipeline, mlflowclient, getattr(pipeline, "_mlflow_run_info"))

    def __init__(
        self,
        pipeline: getml.Pipeline,
        mlflowclient: mlflow.MlflowClient,
        run_info: mlflow.entities.RunInfo,
    ) -> None:
        self._pipeline: getml.Pipeline = pipeline
        self._mlflowclient: mlflow.MlflowClient = mlflowclient
        self._run_info: mlflow.entities.RunInfo = run_info

    def log_parameters(self) -> None:
        parameters: List[Param] = []

        for parameter_name in PARAMETER_NAMES:
            parameter = getattr(self._pipeline, parameter_name)
            if is_dataclass(type(parameter)):
                parameters.extend(self._serialize_dataclass(parameter_name, parameter))
            elif isinstance(parameter, list):
                for id, item in enumerate(parameter):
                    parameters.extend(
                        self._serialize_dataclass(f"{parameter_name}.{id}", item)
                    )
            elif isinstance(parameter, (str, int, float, bool, Literal)):
                parameters.append(Param(parameter_name, str(parameter)))
            else:
                parameters.append(Param(parameter_name, str(parameter)))

        self._mlflowclient.log_batch(run_id=self._run_info.run_id, params=parameters)

    def _serialize_dataclass(self, name: str, parameter: Any) -> List[Param]:
        parameters: List[Param] = []

        current_name: str = parameter.__class__.__name__
        for field in fields(parameter):
            full_field_name: str = f"{name}.{current_name}.{field.name}"
            field_value: Any = getattr(parameter, field.name)
            if is_dataclass(field.type):
                parameters.extend(
                    self._serialize_dataclass(full_field_name, field_value)
                )
            else:
                parameters.append(
                    Param(full_field_name, self._serialize_field_value(field_value))
                )

        return parameters

    def _serialize_field_value(self, field_value: Any) -> str:
        if isinstance(field_value, (frozenset, set)):
            return json.dumps(sorted(field_value))
        if not isinstance(field_value, str):
            return json.dumps(field_value)
        return field_value

    def log_tags(self) -> None:
        tags: List[RunTag] = [RunTag("id", self._pipeline.id)]
        for tag in map(str, self._pipeline.tags):
            if ":" in tag:
                key, value = tag.split(":")
                tags.append(RunTag(key.strip(), value.strip()))
            else:
                tags.append(RunTag(tag, tag))

        self._mlflowclient.log_batch(run_id=self._run_info.run_id, tags=tags)

    def log_metrics(self, run_id: Optional[str] = None) -> None:
        run_id = run_id or self._run_info.run_id
        metrics: List[Metric] = []

        scores = self._pipeline.scores

        if self._pipeline.is_classification:
            metrics.extend(self._serialize_metric("auc", scores.auc, 2))
            metrics.extend(self._serialize_metric("accuracy", scores.accuracy, 2))
            metrics.extend(
                self._serialize_metric("cross_entropy", scores.cross_entropy, 4)
            )

        if self._pipeline.is_regression:
            metrics.extend(self._serialize_metric("mae", scores.mae))
            metrics.extend(self._serialize_metric("rmse", scores.rmse))
            metrics.extend(self._serialize_metric("rsquared", scores.rsquared, 2))

        # TODO: Add feature importance and correlation
        # for feature in pipeline.features:
        #     metrics[f"{feature.name}.importance"] = json.dumps(feature.importance)
        #     metrics[f"{feature.name}.correlation"] = json.dumps(feature.correlation)

        # if len(pipeline.targets) == 1:
        #     metrics["targets"] = getml_pipeline.targets[0]
        # else:
        #     for i, t in enumerate(getml_pipeline.targets):
        #         metrics[f"targets.{i}"] = t

        self._mlflowclient.log_batch(run_id=run_id, metrics=metrics)

    def _serialize_metric(
        self, name: str, values: float | List[float], ndigits: Optional[int] = None
    ) -> List[Metric]:
        timestamp: int = get_current_time_millis()
        if isinstance(values, list):
            return [
                Metric(
                    key=f"name.{id}",
                    value=self.maybe_round(value, ndigits),
                    timestamp=timestamp,
                    step=0,
                )
                for id, value in enumerate(values)
            ]
        return [
            Metric(
                key=name,
                value=self.maybe_round(values, ndigits),
                timestamp=timestamp,
                step=0,
            )
        ]

    @staticmethod
    def maybe_round(value: float, ndigits: Optional[int]) -> float:
        return value if ndigits is None else round(value, ndigits)
