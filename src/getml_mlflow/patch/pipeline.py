from __future__ import annotations

import getpass
import socket
from types import TracebackType
from typing import Callable, Dict, Optional, Sequence, Type, Union

import getml
import mlflow
import mlflow.entities
import numpy
from getml.data import DataFrame, Subset
from getml.pipeline import Pipeline, Scores
from mlflow import MlflowClient
from mlflow.entities import RunStatus, RunTag
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_USER
from numpy.typing import NDArray

from getml_mlflow.constants import DEFAULT_GETML_PROJECTS_PATH
from getml_mlflow.data.dataframelike import DataFrameLike
from getml_mlflow.logging.function import FunctionLogger
from getml_mlflow.logging.logger import log_exit_exception
from getml_mlflow.logging.pipeline import PipelineLogger
from getml_mlflow.logging.systemmetrics import SystemMetricsLogger
from getml_mlflow.loggingconfiguration import LoggingConfiguration


class Run:
    def __init__(
        self,
        mlflow_client: MlflowClient,
        pipeline: Pipeline,
        name: str,
        *,
        create_runs: bool = True,
        extra_tags: Optional[Dict[str, str]] = None,
    ) -> None:
        self._pipeline: Pipeline = pipeline
        self._mlflow_client: MlflowClient = mlflow_client
        self._run: Optional[mlflow.entities.Run] = None
        self._name: str = name
        self._create_runs: bool = create_runs
        self._extra_tags: Dict[str, str] = extra_tags or {}

    def __enter__(self) -> Run:
        if not self._create_runs:
            self._run = mlflow.active_run()
            if self._run is None:
                raise RuntimeError("No active MLflow run found.")
            self._log_extra_tags()
            return self

        create_run_args: dict = {
            "experiment_id": self._experiment_id(),
            "run_name": self._name,
            "tags": {
                "id": self._pipeline.id,
                MLFLOW_USER: f"{getpass.getuser()}@{socket.gethostname()}",
            },
        }
        if parent_run_id := self._parent_run_id():
            create_run_args["tags"].update(
                {
                    MLFLOW_PARENT_RUN_ID: parent_run_id,
                }
            )
        self._run = self._mlflow_client.create_run(**create_run_args)
        self._log_extra_tags()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
    ) -> None:
        if exc_type is not None and exc_value is not None:
            log_exit_exception(
                self._mlflow_client, self.id, exc_type, exc_value, exc_traceback
            )
            if self._create_runs:
                self._mlflow_client.set_terminated(
                    self.id, status=RunStatus.to_string(RunStatus.FAILED)
                )
        else:
            if self._create_runs:
                self._mlflow_client.set_terminated(
                    self.id, status=RunStatus.to_string(RunStatus.FINISHED)
                )
        self._run = None

    def _experiment_id(self) -> str:
        if run_info := getattr(self._pipeline, "_mlflow_run_info"):
            return run_info.experiment_id
        else:
            project_name: str = getml.project.name
            if experiment := self._mlflow_client.get_experiment_by_name(project_name):
                return experiment.experiment_id

            raise LookupError(f"MLflow Experiment '{project_name}' not found")

    def _parent_run_id(self) -> Optional[str]:
        if run_info := getattr(self._pipeline, "_mlflow_run_info"):
            return run_info.run_id

        return None

    @property
    def id(self) -> str:
        assert self._run, "RUN is missing. Make sure to be inside a context manager."
        return self._run.info.run_id

    @property
    def info(self) -> mlflow.entities.RunInfo:
        assert self._run, "RUN is missing. Make sure to be inside a context manager."
        return self._run.info

    def _log_extra_tags(self) -> None:
        if self._extra_tags:
            self._mlflow_client.log_batch(
                self.id,
                tags=[RunTag(*item) for item in self._extra_tags.items()],
            )

    @property
    def run(self) -> mlflow.entities.Run:
        assert self._run, "RUN is missing. Make sure to be inside a context manager."
        return self._run


def init(original: Callable, pipeline: Pipeline, *args, **kwargs) -> None:
    init_method: Callable = original
    if not hasattr(pipeline, "_mlflow_run_info"):
        setattr(pipeline, "_mlflow_run_info", None)

    init_method(pipeline, *args, **kwargs)


def load(
    original: Callable,
    name: str,
    *,
    logging_configuration: LoggingConfiguration = LoggingConfiguration(),
) -> Pipeline:
    mlflow_client: MlflowClient = logging_configuration.general.mlflow_client
    load_function: Callable = original

    load_output: Pipeline = load_function(name)

    with Run(
        mlflow_client=mlflow_client,
        pipeline=load_output,
        name=pipeline_name(load_output),
        create_runs=logging_configuration.general.create_runs,
        extra_tags=logging_configuration.general.extra_tags,
    ) as run:
        setattr(load_output, "_mlflow_run_info", run.info)

    return load_output


def pipeline_name(pipeline: Pipeline) -> str:
    return f"Pipeline-{pipeline.id.replace(' ', '_')}"


def fit(
    original: Callable,
    pipeline: Pipeline,
    population_table: Union[DataFrameLike, Subset],
    peripheral_tables: Optional[
        Union[
            Sequence[DataFrameLike],
            dict[str, DataFrameLike],
        ]
    ] = None,
    validation_table: Optional[Union[DataFrameLike, Subset]] = None,
    check: bool = True,
    *,
    logging_configuration: LoggingConfiguration = LoggingConfiguration(),
) -> Pipeline:
    mlflow_client: MlflowClient = logging_configuration.general.mlflow_client
    fit_method: Callable = original

    ###############
    # Log pipeline
    with Run(
        mlflow_client=mlflow_client,
        pipeline=pipeline,
        name=pipeline_name(pipeline),
        create_runs=logging_configuration.general.create_runs,
        extra_tags=logging_configuration.general.extra_tags,
    ) as run:
        setattr(pipeline, "_mlflow_run_info", run.info)
        pipeline_run_id: str = run.info.run_id
        pipeline_logger: PipelineLogger = PipelineLogger(
            mlflow_client,
            run.id,
            pipeline,
            logging_configuration=logging_configuration.pipeline,
            getml_project_path=logging_configuration.general.getml_project_path
            or DEFAULT_GETML_PROJECTS_PATH,
        )
        pipeline_logger.log_constructor_arguments()

    ###############
    # Log fit call
    with Run(
        mlflow_client=mlflow_client,
        pipeline=pipeline,
        name="fit",
        create_runs=logging_configuration.general.create_runs,
        extra_tags=logging_configuration.general.extra_tags,
    ) as fit_run:
        with PipelineLogger(
            mlflow_client,
            fit_run.id,
            pipeline,
            logging_configuration=logging_configuration.pipeline,
            getml_project_path=logging_configuration.general.getml_project_path
            or DEFAULT_GETML_PROJECTS_PATH,
        ):
            with SystemMetricsLogger(
                mlflow_client,
                fit_run.id,
                log_system_metrics=logging_configuration.general.log_system_metrics,
            ):
                fit_output: Pipeline = FunctionLogger(
                    mlflow_client,
                    fit_run.run,
                    pipeline,
                    function_logging_configuration=logging_configuration.function,
                    data_container_logging_configuration=logging_configuration.data_container,
                ).log(fit_method)(
                    pipeline,
                    population_table,
                    peripheral_tables,
                    validation_table,
                    check,
                )
            mlflow_client.set_tag(fit_run.id, "id", pipeline.id)

    ##################
    # Update pipeline
    mlflow_client.set_tag(pipeline_run_id, "id", pipeline.id)
    if logging_configuration.general.create_runs:
        mlflow_client.update_run(
            run_id=pipeline_run_id,
            name=pipeline_name(pipeline),
        )

    return fit_output


def score(
    original: Callable,
    pipeline: Pipeline,
    population_table: Union[DataFrameLike, Subset],
    peripheral_tables: Optional[
        Union[
            Sequence[DataFrameLike],
            Dict[str, DataFrameLike],
        ]
    ] = None,
    *,
    logging_configuration: LoggingConfiguration = LoggingConfiguration(),
) -> Scores:
    score_method: Callable = original
    mlflow_client: MlflowClient = logging_configuration.general.mlflow_client

    with Run(
        mlflow_client=mlflow_client,
        pipeline=pipeline,
        name="score",
        create_runs=logging_configuration.general.create_runs,
        extra_tags=logging_configuration.general.extra_tags,
    ) as score_run:
        with PipelineLogger(
            mlflow_client,
            score_run.id,
            pipeline,
            logging_configuration=logging_configuration.pipeline,
            getml_project_path=logging_configuration.general.getml_project_path
            or DEFAULT_GETML_PROJECTS_PATH,
        ):
            score_output: Scores = FunctionLogger(
                mlflow_client,
                score_run.run,
                pipeline,
                function_logging_configuration=logging_configuration.function,
                data_container_logging_configuration=logging_configuration.data_container,
            ).log(score_method)(pipeline, population_table, peripheral_tables)

    return score_output


def predict(
    original: Callable,
    pipeline: Pipeline,
    population_table: Union[DataFrameLike, Subset],
    peripheral_tables: Optional[
        Union[
            Sequence[DataFrameLike],
            Dict[str, DataFrameLike],
        ]
    ] = None,
    table_name: str = "",
    *,
    logging_configuration: LoggingConfiguration = LoggingConfiguration(),
) -> Union[NDArray[numpy.float_], None]:
    mlflow_client: MlflowClient = logging_configuration.general.mlflow_client
    predict_method: Callable = original

    with Run(
        mlflow_client=mlflow_client,
        pipeline=pipeline,
        name="predict",
        create_runs=logging_configuration.general.create_runs,
        extra_tags=logging_configuration.general.extra_tags,
    ) as predict_run:
        with PipelineLogger(
            mlflow_client=mlflow_client,
            run_id=predict_run.id,
            pipeline=pipeline,
            logging_configuration=logging_configuration.pipeline,
            getml_project_path=logging_configuration.general.getml_project_path
            or DEFAULT_GETML_PROJECTS_PATH,
        ):
            predict_output: Union[NDArray[numpy.float_], None] = FunctionLogger(
                mlflow_client,
                predict_run.run,
                pipeline,
                function_logging_configuration=logging_configuration.function,
                data_container_logging_configuration=logging_configuration.data_container,
            ).log(
                predict_method
            )(pipeline, population_table, peripheral_tables, table_name)

    return predict_output


def transform(
    original: Callable,
    pipeline: Pipeline,
    population_table: Union[DataFrameLike, Subset],
    peripheral_tables: Optional[
        Union[
            Sequence[DataFrameLike],
            Dict[str, DataFrameLike],
        ]
    ] = None,
    df_name: str = "",
    table_name: str = "",
    *,
    logging_configuration: LoggingConfiguration = LoggingConfiguration(),
) -> Union[DataFrame, NDArray[numpy.float_], None]:
    mlflow_client: MlflowClient = logging_configuration.general.mlflow_client
    transform_method: Callable = original

    with Run(
        mlflow_client=mlflow_client,
        pipeline=pipeline,
        name="transform",
        create_runs=logging_configuration.general.create_runs,
        extra_tags=logging_configuration.general.extra_tags,
    ) as transform_run:
        with PipelineLogger(
            mlflow_client,
            transform_run.id,
            pipeline,
            logging_configuration=logging_configuration.pipeline,
            getml_project_path=logging_configuration.general.getml_project_path
            or DEFAULT_GETML_PROJECTS_PATH,
        ):
            transform_output: Union[DataFrame, NDArray[numpy.float_], None] = (
                FunctionLogger(
                    mlflow_client,
                    transform_run.run,
                    pipeline,
                    function_logging_configuration=logging_configuration.function,
                    data_container_logging_configuration=logging_configuration.data_container,
                ).log(transform_method)(
                    pipeline, population_table, peripheral_tables, df_name, table_name
                )
            )

    return transform_output
