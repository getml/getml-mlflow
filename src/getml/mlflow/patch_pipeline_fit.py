from typing import Callable, Optional, Sequence, Union

from typing_extensions import override

import getml
import mlflow
from getml.mlflow.table_logging import TableLogging
from getml.mlflow.systemmetrics import SystemMetrics
from mlflow.utils.autologging_utils.safety import PatchFunction
from getml.mlflow.pipeline_logging import PipelineLogging


class PatchPipelineFit(PatchFunction):
    @override
    def _patch_implementation(
        self,
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
        # TODO: Add tags
        # TODO: Add description
        pipeline_run_id = getattr(pipeline, "mlflow_run_id")
        with mlflow.start_run(
            run_name="fit",
            nested=True,
            parent_run_id=pipeline_run_id,
        ) as fit_run:
            run_id = fit_run.info.run_id

            TableLogging.log_table(population_table, "Population")
            if peripheral_tables is not None:
                TableLogging.log_peripheral_tables(peripheral_tables)
            if validation_table is not None:
                TableLogging.log_table(validation_table, "Validation")

            with SystemMetrics(run_id):
                fit_output: getml.Pipeline = fit_method(
                    pipeline,
                    population_table,
                    peripheral_tables,
                    validation_table,
                    check,
                )

            PipelineLogging.log_metrics(pipeline, run_id)

            return fit_output
