from types import TracebackType
from typing import Callable, Dict, Optional, Sequence, Type, Union

import getml
import mlflow
import mlflow.entities
import numpy
from mlflow.entities import Param, RunStatus, RunTag
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from numpy.typing import NDArray

from getml_mlflow.data.dataframelike import DataFrameLike
from getml_mlflow.logging.datacontainer import DataContainerLogger
from getml_mlflow.logging.logger import log_exit_exception
from getml_mlflow.logging.numpy import NumpyLogger
from getml_mlflow.logging.pipeline import PipelineLogger
from getml_mlflow.logging.systemmetrics import SystemMetricsLogger
from getml_mlflow.loggingconfiguration import LoggingConfiguration


class Run:
    def __init__(
        self,
        mlflowclient: mlflow.MlflowClient,
        pipeline: getml.Pipeline,
        name: str,
        *,
        create_runs: bool = True,
        extra_tags: Optional[Dict[str, str]] = None,
    ) -> None:
        self._pipeline: getml.Pipeline = pipeline
        self._mlflowclient: mlflow.MlflowClient = mlflowclient
        self._run: Optional[mlflow.entities.Run] = None
        self._name: str = name
        self._create_runs: bool = create_runs
        self._extra_tags: Dict[str, str] = extra_tags or {}

    def __enter__(self) -> "Run":
        if not self._create_runs:
            self._run = mlflow.active_run()
            if self._run is None:
                raise RuntimeError("No active MLflow run found.")
            self._log_extra_tags()
            return self

        create_run_args: dict = {
            "experiment_id": self._experiment_id(),
            "run_name": self._name,
            "tags": {"id": self._pipeline.id},
        }
        if parent_run_id := self._parent_run_id():
            create_run_args["tags"].update(
                {
                    MLFLOW_PARENT_RUN_ID: parent_run_id,
                }
            )
        self._run = self._mlflowclient.create_run(**create_run_args)
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
                self._mlflowclient, self.id, exc_type, exc_value, exc_traceback
            )
            if self._create_runs:
                self._mlflowclient.set_terminated(
                    self.id, status=RunStatus.to_string(RunStatus.FAILED)
                )
        else:
            if self._create_runs:
                self._mlflowclient.set_terminated(
                    self.id, status=RunStatus.to_string(RunStatus.FINISHED)
                )
        self._run = None

    def _experiment_id(self) -> str:
        if run_info := getattr(self._pipeline, "_mlflow_run_info"):
            return run_info.experiment_id
        else:
            project_name: str = getml.project.name
            if experiment := self._mlflowclient.get_experiment_by_name(project_name):
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

    def _log_extra_tags(self):
        if self._extra_tags:
            self._mlflowclient.log_batch(
                self.id,
                tags=[RunTag(*item) for item in self._extra_tags.items()],
            )


def init(original: Callable, pipeline: getml.Pipeline, *args, **kwargs):
    init_method: Callable = original

    if not hasattr(pipeline, "_mlflow_run_info"):
        setattr(pipeline, "_mlflow_run_info", None)

    init_method(pipeline, *args, **kwargs)


def pipeline_name(pipeline: getml.Pipeline) -> str:
    return "Pipeline-{}".format(pipeline.id.replace(" ", "_"))


def fit(
    original: Callable,
    pipeline: getml.Pipeline,
    population_table: Union[DataFrameLike, getml.data.Subset],
    peripheral_tables: Optional[
        Union[
            Sequence[DataFrameLike],
            dict[str, DataFrameLike],
        ]
    ] = None,
    validation_table: Optional[Union[DataFrameLike, getml.data.Subset]] = None,
    check: bool = True,
    *,
    logging_configuration: LoggingConfiguration = LoggingConfiguration(),
) -> getml.Pipeline:
    mlflowclient = logging_configuration.mlflow_client
    fit_method: Callable = original

    with Run(
        mlflowclient=mlflowclient,
        pipeline=pipeline,
        name=pipeline_name(pipeline),
        create_runs=logging_configuration.create_runs,
        extra_tags=logging_configuration.extra_tags,
    ) as run:
        setattr(pipeline, "_mlflow_run_info", run.info)
        pipeline_logger: PipelineLogger = PipelineLogger(
            mlflowclient,
            run.id,
            pipeline,
            log_parameters=logging_configuration.log_pipeline_parameters,
            log_tags=logging_configuration.log_pipeline_tags,
            log_scores=logging_configuration.log_pipeline_scores,
            log_features=logging_configuration.log_pipeline_features,
            log_columns=logging_configuration.log_pipeline_columns,
            log_targets=logging_configuration.log_pipeline_targets,
        )
        pipeline_logger.log_constructor_arguments()

        # TODO: log data model
        #

        with Run(
            mlflowclient=mlflowclient,
            pipeline=pipeline,
            name="fit",
            create_runs=logging_configuration.create_runs,
            extra_tags=logging_configuration.extra_tags,
        ) as fit_run:
            with PipelineLogger(
                mlflowclient,
                fit_run.id,
                pipeline,
                log_parameters=logging_configuration.log_pipeline_parameters,
                log_tags=logging_configuration.log_pipeline_tags,
                log_scores=logging_configuration.log_pipeline_scores,
                log_features=logging_configuration.log_pipeline_features,
                log_columns=logging_configuration.log_pipeline_columns,
                log_targets=logging_configuration.log_pipeline_targets,
            ):
                data_container_logger: DataContainerLogger = DataContainerLogger.as_input(
                    mlflowclient,
                    fit_run.id,
                    log_information=logging_configuration.log_data_container_information,
                    log_as_artifact=logging_configuration.log_data_container_as_artifact,
                )
                if logging_configuration.log_function_parameters:
                    data_container_logger.log_data_container(
                        population_table, "Population"
                    )
                if (
                    logging_configuration.log_function_parameters
                    and peripheral_tables is not None
                ):
                    data_container_logger.log_data_containers(
                        peripheral_tables, "Peripheral"
                    )
                if (
                    logging_configuration.log_function_parameters
                    and validation_table is not None
                ):
                    data_container_logger.log_data_container(
                        validation_table, "Validation"
                    )

                with SystemMetricsLogger(
                    mlflowclient,
                    fit_run.id,
                    log_system_metrics=logging_configuration.log_system_metrics,
                ):
                    fit_output: getml.Pipeline = fit_method(
                        pipeline,
                        population_table,
                        peripheral_tables,
                        validation_table,
                        check,
                    )
                mlflowclient.set_tag(fit_run.id, "id", pipeline.id)

        mlflowclient.set_tag(run.id, "id", pipeline.id)
        if logging_configuration.create_runs:
            mlflowclient.update_run(
                run_id=run.id,
                name=pipeline_name(pipeline),
            )

    return fit_output


def score(
    original: Callable,
    pipeline: getml.Pipeline,
    population_table: Union[DataFrameLike, getml.data.Subset],
    peripheral_tables: Optional[
        Union[
            Sequence[DataFrameLike],
            Dict[str, DataFrameLike],
        ]
    ] = None,
    *,
    logging_configuration: LoggingConfiguration = LoggingConfiguration(),
):
    score_method: Callable = original
    mlflowclient = logging_configuration.mlflow_client

    with Run(
        mlflowclient=mlflowclient,
        pipeline=pipeline,
        name="score",
        create_runs=logging_configuration.create_runs,
        extra_tags=logging_configuration.extra_tags,
    ) as score_run:
        with PipelineLogger(
            mlflowclient,
            score_run.id,
            pipeline,
            log_parameters=logging_configuration.log_pipeline_parameters,
            log_tags=logging_configuration.log_pipeline_tags,
            log_scores=logging_configuration.log_pipeline_scores,
            log_features=logging_configuration.log_pipeline_features,
            log_columns=logging_configuration.log_pipeline_columns,
            log_targets=logging_configuration.log_pipeline_targets,
        ):
            data_container_logger: DataContainerLogger = DataContainerLogger.as_input(
                mlflowclient,
                score_run.id,
                log_information=logging_configuration.log_data_container_information,
                log_as_artifact=logging_configuration.log_data_container_as_artifact,
            )
            if logging_configuration.log_function_parameters:
                data_container_logger.log_data_container(population_table, "Population")
            if (
                logging_configuration.log_function_parameters
                and peripheral_tables is not None
            ):
                data_container_logger.log_data_containers(
                    peripheral_tables, "Peripheral"
                )

            score_output: getml.pipeline.Scores = score_method(
                pipeline, population_table, peripheral_tables
            )

    return score_output


def predict(
    original: Callable,
    pipeline: getml.Pipeline,
    population_table: Union[DataFrameLike, getml.data.Subset],
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
    mlflowclient = logging_configuration.mlflow_client
    predict_method: Callable = original

    with Run(
        mlflowclient=mlflowclient,
        pipeline=pipeline,
        name="predict",
        create_runs=logging_configuration.create_runs,
        extra_tags=logging_configuration.extra_tags,
    ) as predict_run:
        with PipelineLogger(
            mlflowclient=mlflowclient,
            run_id=predict_run.id,
            pipeline=pipeline,
            log_parameters=logging_configuration.log_pipeline_parameters,
            log_tags=logging_configuration.log_pipeline_tags,
            log_scores=logging_configuration.log_pipeline_scores,
            log_features=logging_configuration.log_pipeline_features,
            log_columns=logging_configuration.log_pipeline_columns,
            log_targets=logging_configuration.log_pipeline_targets,
        ):
            data_container_logger: DataContainerLogger = DataContainerLogger.as_input(
                mlflowclient,
                predict_run.id,
                log_information=logging_configuration.log_data_container_information,
                log_as_artifact=logging_configuration.log_data_container_as_artifact,
            )
            if logging_configuration.log_function_parameters:
                data_container_logger.log_data_container(population_table, "Population")
            if (
                logging_configuration.log_function_parameters
                and peripheral_tables is not None
            ):
                data_container_logger.log_data_containers(
                    peripheral_tables, "Peripheral"
                )

            if logging_configuration.log_function_parameters:
                mlflowclient.log_param(
                    run_id=predict_run.id, key="table_name", value=table_name
                )

            predict_output: Union[NDArray[numpy.float_], None] = predict_method(
                pipeline, population_table, peripheral_tables, table_name
            )
            if logging_configuration.log_function_return and predict_output is not None:
                NumpyLogger(mlflowclient, predict_run.id).log_ndarray_as_artifact(
                    data=predict_output,
                    name="predict_output",
                )

    return predict_output


def transform(
    original: Callable,
    pipeline: getml.Pipeline,
    population_table: Union[DataFrameLike, getml.data.Subset],
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
) -> Union[getml.DataFrame, NDArray[numpy.float_], None]:
    mlflowclient = logging_configuration.mlflow_client
    transform_method: Callable = original

    with Run(
        mlflowclient=mlflowclient,
        pipeline=pipeline,
        name="transform",
        create_runs=logging_configuration.create_runs,
        extra_tags=logging_configuration.extra_tags,
    ) as transform_run:
        with PipelineLogger(
            mlflowclient,
            transform_run.id,
            pipeline,
            log_parameters=logging_configuration.log_pipeline_parameters,
            log_tags=logging_configuration.log_pipeline_tags,
            log_scores=logging_configuration.log_pipeline_scores,
            log_features=logging_configuration.log_pipeline_features,
            log_columns=logging_configuration.log_pipeline_columns,
            log_targets=logging_configuration.log_pipeline_targets,
        ):
            data_container_logger: DataContainerLogger = DataContainerLogger.as_input(
                mlflowclient,
                transform_run.id,
                log_information=logging_configuration.log_data_container_information,
                log_as_artifact=logging_configuration.log_data_container_as_artifact,
            )
            if logging_configuration.log_function_parameters:
                data_container_logger.log_data_container(population_table, "Population")
            if (
                logging_configuration.log_function_parameters
                and peripheral_tables is not None
            ):
                data_container_logger.log_data_containers(
                    peripheral_tables, "Peripheral"
                )
            if logging_configuration.log_function_parameters:
                mlflowclient.log_batch(
                    transform_run.id,
                    params=[Param("df_name", df_name), Param("table_name", table_name)],
                )

            transform_output: Union[getml.DataFrame, NDArray[numpy.float_], None] = (
                transform_method(
                    pipeline, population_table, peripheral_tables, df_name, table_name
                )
            )
            if (
                logging_configuration.log_function_return
                and transform_output is not None
            ):
                if isinstance(transform_output, getml.DataFrame):
                    DataContainerLogger.as_artifact(
                        mlflowclient,
                        transform_run.id,
                        log_information=logging_configuration.log_data_container_information,
                        log_as_artifact=logging_configuration.log_data_container_as_artifact,
                    ).log_data_container(
                        data_container=transform_output,
                        context="output",
                    )
                elif isinstance(transform_output, numpy.ndarray):
                    NumpyLogger(mlflowclient, transform_run.id).log_ndarray_as_artifact(
                        data=transform_output,
                        name="output",
                        artifact_path="output",
                    )

    return transform_output
