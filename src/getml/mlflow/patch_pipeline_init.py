from typing import Callable, Optional, Sequence, Union

from typing_extensions import override

import getml
import mlflow
from getml.mlflow.systemmetrics import SystemMetrics
from mlflow.utils.autologging_utils.safety import PatchFunction

from getml.mlflow.pipeline_logging import PipelineLogging


class PatchPipelineInit(PatchFunction):
    @override
    def _patch_implementation(
        self, original: Callable, pipeline: getml.Pipeline, *args, **kwargs
    ):
        init_method: Callable = original

        active_run = mlflow.active_run()
        # TODO: Add tags
        # TODO: Add description
        if hasattr(pipeline, "mlflow_run_id"):
            init_method(pipeline, *args, **kwargs)
        else:
            with mlflow.start_run(
                run_name="Pipeline",
                nested=False if active_run is None else True,
                parent_run_id=None if active_run is None else active_run.info.run_id,
            ) as pipeline_run:
                run_id = pipeline_run.info.run_id
                setattr(pipeline, "mlflow_run_id", run_id)

                init_method(pipeline, *args, **kwargs)

                PipelineLogging.log_parameters(pipeline, run_id)
                PipelineLogging.log_tags(pipeline)

                # TODO: predict
                # TODO: transform
