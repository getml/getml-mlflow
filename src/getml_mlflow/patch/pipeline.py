from typing import Callable, Dict, Optional, Sequence, Union

import getml
import mlflow
import mlflow.entities
from mlflow.entities import RunStatus

from getml_mlflow import logging
from getml_mlflow.logging.systemmetrics import SystemMetrics


def init(original: Callable, pipeline: getml.Pipeline, *args, **kwargs):
    init_method: Callable = original

    if not hasattr(pipeline, "_mlflow_run_info"):
        setattr(pipeline, "_mlflow_run_info", None)

    init_method(pipeline, *args, **kwargs)


# TODO: predict
# TODO: transform


class PipelineRun:
    __slots__ = ["_pipeline", "_mlflowclient", "_run", "_experiment"]

    def __init__(
        self, _pipeline: getml.Pipeline, mlflowclient: mlflow.MlflowClient
    ) -> None:
        self._pipeline: getml.Pipeline = _pipeline
        self._mlflowclient: mlflow.MlflowClient = mlflowclient
        self._run: Optional[mlflow.entities.Run] = None
        experiment: Optional[mlflow.entities.Experiment] = (
            mlflow.get_experiment_by_name(getml.project.name)
        )
        if experiment is None:
            raise LookupError(f"MLflow Experiment '{getml.project.name}' not found")
        self._experiment: mlflow.entities.Experiment = experiment

    def __enter__(self) -> "PipelineRun":
        self._run = self._mlflowclient.create_run(
            experiment_id=self._experiment.experiment_id, run_name=self._run_name()
        )
        setattr(self._pipeline, "_mlflow_run_info", self._run.info)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._mlflowclient.update_run(
            run_id=self.id, status="FINISHED", name=self._run_name()
        )
        self._run = None

    def _run_name(self) -> str:
        return "Pipeline-{}".format(self._pipeline.id.replace(" ", "_"))

    @property
    def id(self) -> str:
        assert self._run, "RUN is missing. Make sure to be inside a context manager."
        return self._run.info.run_id

    @property
    def run(self) -> mlflow.entities.Run:
        assert self._run, "RUN is missing. Make sure to be inside a context manager."
        return self._run


class PipelineSubRun:
    __slots__ = ["_mlflowclient", "_run", "_run_name", "_parent_run_info"]

    def __init__(
        self,
        parent_run_info: mlflow.entities.RunInfo,
        run_name: str,
        mlflowclient: mlflow.MlflowClient,
    ) -> None:
        self._parent_run_info: mlflow.entities.RunInfo = parent_run_info
        self._mlflowclient: mlflow.MlflowClient = mlflowclient
        self._run_name: str = run_name
        self._run: Optional[mlflow.entities.Run] = None

    def __enter__(self) -> "PipelineSubRun":
        self._run = self._mlflowclient.create_run(
            experiment_id=self._parent_run_info.experiment_id,
            tags={"mlflow.parentRunId": self._parent_run_info.run_id},
            run_name=self._run_name,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._mlflowclient.set_terminated(
            self.id, status=RunStatus.to_string(RunStatus.FINISHED)
        )
        self._run = None

    @property
    def id(self) -> str:
        assert self._run, "RUN is missing. Make sure to be inside a context manager."
        return self._run.info.run_id


def fit(
    original: Callable,
    pipeline: getml.Pipeline,
    population_table: Union[getml.DataFrame, getml.data.View, getml.data.Subset],
    peripheral_tables: Optional[
        Union[
            Sequence[Union[getml.DataFrame, getml.data.View]],
            dict[str, Union[getml.DataFrame, getml.data.View]],
        ]
    ] = None,
    validation_table: Optional[
        Union[getml.DataFrame, getml.data.View, getml.data.Subset]
    ] = None,
    check: bool = True,
    mlflowclient: Optional[mlflow.MlflowClient] = None,
):
    mlflowclient = mlflowclient or mlflow.MlflowClient()
    fit_method: Callable = original

    with PipelineRun(_pipeline=pipeline, mlflowclient=mlflowclient) as pipeline_run:
        with PipelineSubRun(
            parent_run_info=pipeline_run.run.info,
            run_name="fit",
            mlflowclient=mlflowclient,
        ) as fit_run:
            logging.table.log_table(population_table, "Population")
            if peripheral_tables is not None:
                logging.table.log_peripheral_tables(peripheral_tables)
            if validation_table is not None:
                logging.table.log_table(validation_table, "Validation")

            with SystemMetrics(fit_run.id):
                fit_output: getml.Pipeline = fit_method(
                    pipeline,
                    population_table,
                    peripheral_tables,
                    validation_table,
                    check,
                )

            logging.pipeline.log_metrics(pipeline, fit_run.id)
            mlflowclient.set_tag(fit_run.id, "id", pipeline.id)

        mlflowclient.set_tag(pipeline_run.id, "id", pipeline.id)
        logging.pipeline.log_parameters(pipeline, pipeline_run.id)
        logging.pipeline.log_tags(pipeline)

    return fit_output


def score(
    original: Callable,
    pipeline: getml.Pipeline,
    population_table: Union[getml.DataFrame, getml.data.View, getml.data.Subset],
    peripheral_tables: Optional[
        Union[
            Sequence[Union[getml.DataFrame, getml.data.View]],
            Dict[str, Union[getml.DataFrame, getml.data.View]],
        ]
    ] = None,
    mlflowclient: Optional[mlflow.MlflowClient] = None,
):
    score_method: Callable = original
    mlflowclient = mlflowclient or mlflow.MlflowClient()

    with PipelineSubRun(
        parent_run_info=getattr(pipeline, "_mlflow_run_info"),
        run_name="score",
        mlflowclient=mlflowclient,
    ) as score_run:
        logging.table.log_table(population_table, "Population")
        if peripheral_tables is not None:
            logging.table.log_peripheral_tables(peripheral_tables)

        score_output: getml.pipeline.Scores = score_method(
            pipeline, population_table, peripheral_tables
        )
        logging.pipeline.log_metrics(pipeline, score_run.id)
        mlflowclient.set_tag(score_run.id, "id", pipeline.id)

    return score_output


# def predict(
#     original: Callable,
#     pipeline: getml.Pipeline,
#     population_table: Union[getml.DataFrame, getml.data.View, getml.data.Subset],
#     peripheral_tables: Optional[
#         Union[
#             Sequence[Union[getml.DataFrame, getml.data.View]],
#             Dict[str, Union[getml.DataFrame, getml.data.View]],
#         ]
#     ] = None,
#     table_name: str = "",
# ) -> Union[NDArray[numpy.float_], None]:
#     predict_method: Callable = original
#     pipeline_run_id = getattr(pipeline, "mlflow_run_id")
#
#     # TODO: Add tags
#     # TODO: Add description
#     with mlflow.start_run(
#         run_name="predict",
#         nested=True,
#         parent_run_id=pipeline_run_id,
#     ) as predict_run:
#         run_id = predict_run.info.run_id
#
#         logging.table.log_table(population_table, "Population")
#         if peripheral_tables is not None:
#             logging.table.log_peripheral_tables(peripheral_tables)
#         mlflow.log_param("table_name", table_name)
#
#         predict_output: Union[NDArray[numpy.float_], None] = predict_method(
#             pipeline, population_table, peripheral_tables
#         )
#         if predict_output is not None:
#             logging.numpy.log_ndarray_as_artifact(
#                 predict_output, f"pipeline.{pipeline.id}.predict.npy"
#             )
#         logging.pipeline.log_metrics(pipeline, run_id)
#         mlflow.set_tag(key="id", value=pipeline.id)
#
#         return predict_output
#
#
# def transform(
#     original: Callable,
#     pipeline: getml.Pipeline,
#     population_table: Union[getml.DataFrame, getml.data.View, getml.data.Subset],
#     peripheral_tables: Optional[
#         Union[
#             Sequence[Union[getml.DataFrame, getml.data.View]],
#             Dict[str, Union[getml.DataFrame, getml.data.View]],
#         ]
#     ] = None,
#     df_name: str = "",
#     table_name: str = "",
# ) -> Union[getml.DataFrame, NDArray[numpy.float_], None]:
#     transform_method: Callable = original
#     pipeline_run_id = getattr(pipeline, "mlflow_run_id")
#
#     # TODO: Add tags
#     # TODO: Add description
#     with mlflow.start_run(
#         run_name="transform",
#         nested=True,
#         parent_run_id=pipeline_run_id,
#     ) as transform_run:
#         run_id = transform_run.info.run_id
#
#         logging.table.log_table(population_table, "Population")
#         if peripheral_tables is not None:
#             logging.table.log_peripheral_tables(peripheral_tables)
#         mlflow.log_param("df_name", df_name)
#         mlflow.log_param("table_name", table_name)
#
#         transform_output: Union[getml.DataFrame, NDArray[numpy.float_], None] = (
#             transform_method(pipeline, population_table, peripheral_tables)
#         )
#         if transform_output is not None:
#             if isinstance(transform_output, getml.DataFrame):
#                 logging.table.log_table_as_artifact(transform_output, df_name)
#             elif isinstance(transform_output, numpy.ndarray):
#                 logging.numpy.log_ndarray_as_artifact(
#                     transform_output, f"pipeline.{pipeline.id}.transform.npy"
#                 )
#         logging.pipeline.log_metrics(pipeline, run_id)
#         mlflow.set_tag(key="id", value=pipeline.id)
#
#         return transform_output
#
