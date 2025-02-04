from types import TracebackType
from typing import Callable, Dict, Optional, Sequence, Type, Union

import getml
import mlflow
import mlflow.entities
import numpy
from mlflow.entities import Param, RunStatus
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from numpy.typing import NDArray

from getml_mlflow.data.dataframelike import DataFrameLike
from getml_mlflow.logging.datacontainer import DataContainerLogger
from getml_mlflow.logging.logger import log_exit_exception
from getml_mlflow.logging.numpy import NumpyLogger
from getml_mlflow.logging.pipeline import PipelineLogger
from getml_mlflow.logging.systemmetrics import SystemMetrics


class Run:
    def __init__(
        self, mlflowclient: mlflow.MlflowClient, pipeline: getml.Pipeline, name: str
    ) -> None:
        self._pipeline: getml.Pipeline = pipeline
        self._mlflowclient: mlflow.MlflowClient = mlflowclient
        self._run: Optional[mlflow.entities.Run] = None
        self._name: str = name

    def __enter__(self) -> "Run":
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
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
    ) -> None:
        if exc_type is not None and exc_value is not None:
            log_exit_exception(
                self._mlflowclient, None, exc_type, exc_value, exc_traceback
            )
            self._mlflowclient.set_terminated(
                self.id, status=RunStatus.to_string(RunStatus.FAILED)
            )
        else:
            self._mlflowclient.set_terminated(
                self.id, status=RunStatus.to_string(RunStatus.FINISHED)
            )

    def _experiment_id(self) -> str:
        if run_info := getattr(self._pipeline, "_mlflow_run_info"):
            return run_info.experiment_id
        else:
            project_name: str = getml.project.name
            if experiment := mlflow.get_experiment_by_name(project_name):
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
    mlflowclient: Optional[mlflow.MlflowClient] = None,
) -> getml.Pipeline:
    mlflowclient = mlflowclient or mlflow.MlflowClient()
    fit_method: Callable = original

    with Run(
        mlflowclient=mlflowclient, pipeline=pipeline, name=pipeline_name(pipeline)
    ) as run:
        setattr(pipeline, "_mlflow_run_info", run.info)
        pipeline_logger: PipelineLogger = PipelineLogger(mlflowclient, run.id, pipeline)
        pipeline_logger.log_given_information()
        # TODO: log data model
        #

        with Run(mlflowclient=mlflowclient, pipeline=pipeline, name="fit") as fit_run:
            with PipelineLogger(mlflowclient, fit_run.id, pipeline):
                data_container_logger: DataContainerLogger = (
                    DataContainerLogger.as_input(mlflowclient, fit_run.id)
                )
                data_container_logger.log_data_container(population_table, "Population")
                if peripheral_tables is not None:
                    data_container_logger.log_data_containers(
                        peripheral_tables, "Peripheral"
                    )
                if validation_table is not None:
                    data_container_logger.log_data_container(
                        validation_table, "Validation"
                    )
                with SystemMetrics(fit_run.id):
                    fit_output: getml.Pipeline = fit_method(
                        pipeline,
                        population_table,
                        peripheral_tables,
                        validation_table,
                        check,
                    )
                mlflowclient.set_tag(fit_run.id, "id", pipeline.id)
        mlflowclient.set_tag(run.id, "id", pipeline.id)
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
    mlflowclient: Optional[mlflow.MlflowClient] = None,
):
    score_method: Callable = original
    mlflowclient = mlflowclient or mlflow.MlflowClient()

    with Run(mlflowclient=mlflowclient, pipeline=pipeline, name="score") as score_run:
        with PipelineLogger(mlflowclient, score_run.id, pipeline):
            data_container_logger: DataContainerLogger = DataContainerLogger.as_input(
                mlflowclient, score_run.id
            )
            data_container_logger.log_data_container(population_table, "Population")
            if peripheral_tables is not None:
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
    mlflowclient: Optional[mlflow.MlflowClient] = None,
) -> Union[NDArray[numpy.float_], None]:
    mlflowclient = mlflowclient or mlflow.MlflowClient()
    predict_method: Callable = original

    with Run(
        mlflowclient=mlflowclient, pipeline=pipeline, name="predict"
    ) as predict_run:
        with PipelineLogger(
            mlflowclient=mlflowclient, run_id=predict_run.id, pipeline=pipeline
        ):
            data_container_logger: DataContainerLogger = DataContainerLogger.as_input(
                mlflowclient, predict_run.id
            )
            data_container_logger.log_data_container(population_table, "Population")
            if peripheral_tables is not None:
                data_container_logger.log_data_containers(
                    peripheral_tables, "Peripheral"
                )

            mlflowclient.log_param(
                run_id=predict_run.id, key="table_name", value=table_name
            )

            predict_output: Union[NDArray[numpy.float_], None] = predict_method(
                pipeline, population_table, peripheral_tables, table_name
            )
            if predict_output is not None:
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
    mlflowclient: Optional[mlflow.MlflowClient] = None,
) -> Union[getml.DataFrame, NDArray[numpy.float_], None]:
    mlflowclient = mlflowclient or mlflow.MlflowClient()
    transform_method: Callable = original

    with Run(
        mlflowclient=mlflowclient,
        pipeline=pipeline,
        name="transform",
    ) as transform_run:
        with PipelineLogger(mlflowclient, transform_run.id, pipeline):
            data_container_logger: DataContainerLogger = DataContainerLogger.as_input(
                mlflowclient, transform_run.id
            )
            data_container_logger.log_data_container(population_table, "Population")
            if peripheral_tables is not None:
                data_container_logger.log_data_containers(
                    peripheral_tables, "Peripheral"
                )
            mlflowclient.log_batch(
                transform_run.id,
                params=[Param("df_name", df_name), Param("table_name", table_name)],
            )

            transform_output: Union[getml.DataFrame, NDArray[numpy.float_], None] = (
                transform_method(
                    pipeline, population_table, peripheral_tables, df_name, table_name
                )
            )
            if transform_output is not None:
                if isinstance(transform_output, getml.DataFrame):
                    DataContainerLogger.as_artifact(
                        mlflowclient, transform_run.id
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
