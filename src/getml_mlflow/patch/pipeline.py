from typing import Callable, Dict, Optional, Sequence, Union

import getml
import mlflow

from getml_mlflow import logging
from getml_mlflow.logging.systemmetrics import SystemMetrics


def init(original: Callable, pipeline: getml.Pipeline, *args, **kwargs):
    init_method: Callable = original

    active_run = mlflow.active_run()
    if hasattr(pipeline, "mlflow_run_id"):
        init_method(pipeline, *args, **kwargs)
    else:
        # TODO: Add tags
        # TODO: Add description
        with mlflow.start_run(
            run_name="Pipeline",
            nested=False if active_run is None else True,
            parent_run_id=None if active_run is None else active_run.info.run_id,
        ) as pipeline_run:
            run_id = pipeline_run.info.run_id
            setattr(pipeline, "mlflow_run_id", run_id)

            init_method(pipeline, *args, **kwargs)

            logging.pipeline.log_parameters(pipeline, run_id)
            logging.pipeline.log_tags(pipeline)


# TODO: predict
# TODO: transform


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
):
    fit_method: Callable = original
    pipeline_run_id = getattr(pipeline, "mlflow_run_id")
    try:
        # TODO: Add tags
        # TODO: Add description
        with mlflow.start_run(
            run_name="fit",
            nested=True,
            parent_run_id=pipeline_run_id,
        ) as fit_run:
            run_id = fit_run.info.run_id

            logging.table.log_table(population_table, "Population")
            if peripheral_tables is not None:
                logging.table.log_peripheral_tables(peripheral_tables)
            if validation_table is not None:
                logging.table.log_table(validation_table, "Validation")

            with SystemMetrics(run_id):
                fit_output: getml.Pipeline = fit_method(
                    pipeline,
                    population_table,
                    peripheral_tables,
                    validation_table,
                    check,
                )

            logging.pipeline.log_metrics(pipeline, run_id)
            mlflow.set_tag(key="id", value=pipeline.id)

            return fit_output
    finally:
        logging.pipeline.set_id_tag(pipeline, pipeline_run_id)


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
):
    score_method: Callable = original
    pipeline_run_id = getattr(pipeline, "mlflow_run_id")
    try:
        # TODO: Add tags
        # TODO: Add description
        with mlflow.start_run(
            run_name="score",
            nested=True,
            parent_run_id=pipeline_run_id,
        ) as score_run:
            run_id = score_run.info.run_id

            logging.table.log_table(population_table, "Population")
            if peripheral_tables is not None:
                logging.table.log_peripheral_tables(peripheral_tables)

            score_output: getml.pipeline.Scores = score_method(
                pipeline, population_table, peripheral_tables
            )
            logging.pipeline.log_metrics(pipeline, run_id)
            mlflow.set_tag(key="id", value=pipeline.id)

            return score_output
    finally:
        logging.pipeline.set_id_tag(pipeline, pipeline_run_id)
