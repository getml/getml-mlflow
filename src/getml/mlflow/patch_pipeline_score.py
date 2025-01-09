from typing import Callable, Dict, Optional, Sequence, Union

from typing_extensions import override

import getml
import mlflow
from getml.mlflow.pipeline_logging import PipelineLogging
from getml.mlflow.table_logging import TableLogging
from mlflow.utils.autologging_utils.safety import PatchFunction


class PatchPipelineScore(PatchFunction):
    @override
    def _patch_implementation(
        self,
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
        # TODO: Add tags
        # TODO: Add description
        pipeline_run_id = getattr(pipeline, "mlflow_run_id")
        with mlflow.start_run(
            run_name="score",
            nested=True,
            parent_run_id=pipeline_run_id,
        ) as score_run:
            run_id = score_run.info.run_id

            TableLogging.log_table(population_table, "Population")
            if peripheral_tables is not None:
                TableLogging.log_peripheral_tables(peripheral_tables)

            # new/
            # target = pipeline.data_model.population.roles.target[0]
            # pop_df = population_table.population.to_pandas()
            # pop_df["predictions"] = pipeline.predict(population_table=population_table, peripheral_tables=peripheral_tables)
            # pop_df["predictions"] = pop_df.round({"predictions": 0})["predictions"].astype(
            #     bool
            # )
            # model_type = ["regressor" if pipeline.is_regression else "classifier"][0]
            # if model_type == "classifier":
            #     pop_df[target] = pop_df[target].astype(bool)

            # mlflow.evaluate(
            #     # model = gm,
            #     data=pop_df,
            #     targets=target,
            #     predictions="predictions",
            #     model_type=["regressor" if pipeline.is_regression else "classifier"][0],
            #     evaluators=["default"],
            # )
            #
            # mlflow.evaluate(
            #     model=None,
            #     data=None,
            #     model_type="classifier" if pipeline.is_classification else "regressor",
            #     targets=None,
            #     predictions=None,
            #     dataset_path=None,
            #     feature_names=None,
            #     evaluators=None,
            #     evaluator_config=None,
            #     custom_metrics=None,
            #     extra_metrics=None,
            #     custom_artifacts=None,
            #     validation_thresholds=None,
            #     baseline_model=None,
            #     env_manager='local',
            #     model_config=None,
            #     baseline_config=None,
            #     inference_params=None
            # )

            # /new

            score_output: getml.pipeline.Scores = score_method(
                pipeline, population_table, peripheral_tables
            )
            # TODO: mlflow.log_metrics(score_output) but should already be done by mlfow.evaluate
            return score_output

            # PipelineLogging.log_metrics(pipeline, run_id)

            # return score_output
