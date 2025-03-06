from __future__ import annotations

import json
import warnings
from dataclasses import is_dataclass
from pathlib import Path
from types import TracebackType
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Protocol,
    Type,
    Union,
    runtime_checkable,
)

from getml.pipeline import Pipeline, Scores
from mlflow import MlflowClient
from mlflow.entities import Metric, Param, RunTag
from mlflow.utils.time import get_current_time_millis
from pydantic import TypeAdapter

from getml_mlflow.constants import DEFAULT_GETML_PROJECTS_PATH
from getml_mlflow.logging.logger import log_exit_exception
from getml_mlflow.loggingconfiguration import PipelineLoggingConfiguration
from getml_mlflow.marshalling.pipeline import log_pipeline_as_artifact
from getml_mlflow.util.callableenum import CallableEnumFactory


@runtime_checkable
class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, Any]]


PipelineParameter = CallableEnumFactory[Callable[[Pipeline], Any]].build(
    "PipelineParameter",
    dict(
        PREPROCESSORS=lambda pipeline: pipeline.preprocessors,
        FEATURE_LEARNERS=lambda pipeline: pipeline.feature_learners,
        FEATURE_SELECTORS=lambda pipeline: pipeline.feature_selectors,
        PREDICTORS=lambda pipeline: pipeline.predictors,
        LOSS_FUNCTION=lambda pipeline: pipeline.loss_function,
        INCLUDE_CATEGORICAL=lambda pipeline: pipeline.include_categorical,
        SHARE_SELECTED_FEATURES=lambda pipeline: pipeline.share_selected_features,
    ),
)


class PipelineLogger:
    def __init__(
        self,
        mlflow_client: MlflowClient,
        run_id: str,
        pipeline: Pipeline,
        *,
        logging_configuration: PipelineLoggingConfiguration = PipelineLoggingConfiguration(),
        getml_project_path: Path = DEFAULT_GETML_PROJECTS_PATH,
    ) -> None:
        self._mlflow_client: MlflowClient = mlflow_client
        self._run_id: str = run_id
        self._pipeline: Pipeline = pipeline
        self._logging_configuration: PipelineLoggingConfiguration = (
            logging_configuration
        )
        self._getml_project_path: Path = getml_project_path

    def log_constructor_arguments(self) -> None:
        self.log_parameters()
        self.log_tags()
        self.log_data_model()

    def log_generated_information(self) -> None:
        self.log_scores()
        self.log_features()
        self.log_columns()
        self.log_targets()
        self.log_pipeline_as_artifact()

    def __enter__(self) -> PipelineLogger:
        self.log_constructor_arguments()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
    ) -> None:
        if exc_type is not None and exc_value is not None:
            log_exit_exception(
                self._mlflow_client, self._run_id, exc_type, exc_value, exc_traceback
            )
        else:
            self.log_generated_information()

    def log_parameters(self) -> None:
        if not self._logging_configuration.log_parameters:
            return

        parameters: List[Param] = []

        for pipeline_parameter in PipelineParameter:
            parameter: Any = pipeline_parameter.value(self._pipeline)
            parameter_name: str = pipeline_parameter.name.lower()
            if isinstance(parameter, list):
                assert all(map(is_dataclass, parameter))
                for id, item in enumerate(parameter):
                    parameters.extend(
                        self._serialize_dataclass(f"{parameter_name}.{id}", item)
                    )
            else:
                parameters.append(Param(parameter_name, str(parameter)))

        self._mlflow_client.log_batch(run_id=self._run_id, params=parameters)

    def _serialize_dataclass(
        self, name: str, parameter: DataclassInstance
    ) -> List[Param]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module="pydantic",
            )
            parameter_as_dict: Dict[str, Any] = TypeAdapter(
                type(parameter)
            ).dump_python(parameter)
        current_name: str = f"{name}.{type(parameter).__name__}"
        return self._to_param_list(
            current_name,
            self._flatten_parameters(parameter_as_dict),
        )

    def _flatten_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        flattened_parameters: Dict[str, Any] = {}
        for key, value in parameters.items():
            if isinstance(value, dict):
                for sub_key, sub_value in self._flatten_parameters(value).items():
                    flattened_parameters[f"{key}.{sub_key}"] = sub_value
            else:
                flattened_parameters[key] = value
        return flattened_parameters

    def _to_param_list(self, name: str, parameters: Dict[str, Any]) -> List[Param]:
        return [
            Param(f"{name}.{key}", self._serialize_field_value(value))
            for key, value in parameters.items()
        ]

    def _serialize_field_value(self, field_value: Any) -> str:
        if isinstance(field_value, (frozenset, set)):
            return json.dumps(sorted(field_value))
        if not isinstance(field_value, str):
            return json.dumps(field_value)
        return field_value

    def log_tags(self) -> None:
        if not self._logging_configuration.log_tags:
            return

        tags: List[RunTag] = [RunTag("id", self._pipeline.id)]
        for tag in map(str, self._pipeline.tags):
            if ":" in tag:
                key, value = tag.split(":")
                tags.append(RunTag(key.strip(), value.strip()))
            else:
                tags.append(RunTag(tag, tag))

        self._mlflow_client.log_batch(run_id=self._run_id, tags=tags)

    def log_scores(self) -> None:
        if not self._logging_configuration.log_scores:
            return

        metrics: List[Metric] = []

        scores: Scores = self._pipeline.scores

        if self._pipeline.is_classification:
            metrics.extend(self._serialize_metric("auc", scores.auc))
            metrics.extend(self._serialize_metric("accuracy", scores.accuracy))
            metrics.extend(
                self._serialize_metric("cross_entropy", scores.cross_entropy)
            )

        if self._pipeline.is_regression:
            metrics.extend(self._serialize_metric("mae", scores.mae))
            metrics.extend(self._serialize_metric("rmse", scores.rmse))
            metrics.extend(self._serialize_metric("rsquared", scores.rsquared))

        self._mlflow_client.log_batch(run_id=self._run_id, metrics=metrics)

    # TODO: Find better way to log and display features
    def log_features(self) -> None:
        if not self._logging_configuration.log_features:
            return

        data: Dict[str, List[Union[int, float, str]]] = {
            "feature-names": [feature.name for feature in self._pipeline.features],
            "feature-importances": [
                feature.importance for feature in self._pipeline.features
            ],
            "feature-correlations": [
                feature.correlation for feature in self._pipeline.features
            ],
        }
        self._mlflow_client.log_table(
            self._run_id, data, f"output/pipeline.{self._pipeline.id}.features.json"
        )

    # TODO: Find better way to log and display columns
    def log_columns(self) -> None:
        if not self._logging_configuration.log_columns:
            return

        data: Dict[str, List[Union[int, float, str]]] = {
            "column-names": [column.name for column in self._pipeline.columns],
            "column-table": [column.table for column in self._pipeline.columns],
            "column-importances": [
                column.importance for column in self._pipeline.columns
            ],
        }
        self._mlflow_client.log_table(
            self._run_id, data, f"output/pipeline.{self._pipeline.id}.columns.json"
        )

    def log_targets(self) -> None:
        if not self._logging_configuration.log_targets:
            return

        self._mlflow_client.log_param(self._run_id, "targets", self._pipeline.targets)

    def _serialize_metric(self, name: str, values: float | List[float]) -> List[Metric]:
        timestamp: int = get_current_time_millis()
        if isinstance(values, list):
            return [
                Metric(
                    key=f"{name}.{id}",
                    value=value,
                    timestamp=timestamp,
                    step=0,
                )
                for id, value in enumerate(values)
            ]
        return [
            Metric(
                key=name,
                value=values,
                timestamp=timestamp,
                step=0,
            )
        ]

    def log_pipeline_as_artifact(self) -> None:
        if not self._logging_configuration.log_as_artifact:
            return

        log_pipeline_as_artifact(
            mlflow_client=self._mlflow_client,
            run_id=self._run_id,
            pipeline=self._pipeline,
            projects_path=self._getml_project_path,
        )

    def log_data_model(self) -> None:
        if not self._logging_configuration.log_data_model:
            return

        self._mlflow_client.log_text(
            run_id=self._run_id,
            text=self._pipeline.data_model._repr_html_(),
            artifact_file="input/data_model.html",
        )
