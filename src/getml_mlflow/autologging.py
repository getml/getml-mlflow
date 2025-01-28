from typing import Dict, Optional

import getml
import mlflow
from mlflow.utils.autologging_utils import autologging_integration
from mlflow.utils.autologging_utils.safety import safe_patch

import getml_mlflow.logging.logger
from getml_mlflow.flavor import FLAVOR_NAME
from getml_mlflow.patch import engine, pipeline


def with_mlflowclient(mlflowclient: mlflow.MlflowClient):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs, mlflowclient=mlflowclient)

        return wrapper

    return decorator


# TODO: Pass and check arguments to the right functions
@autologging_integration(FLAVOR_NAME)
def autolog(
    log_input_examples: bool = False,
    log_model_signatures: bool = True,
    log_models: bool = True,
    log_datasets: bool = True,
    disable: bool = False,
    exclusive: bool = False,
    disable_for_unsupported_versions: bool = False,
    silent: bool = False,
    extra_tags: Optional[Dict[str, str]] = None,
):
    mlflowclient = mlflow.MlflowClient()
    getml_mlflow.logging.logger.set_up()

    safe_patch(
        autologging_integration=FLAVOR_NAME,
        destination=getml.pipeline.Pipeline,
        function_name="__init__",
        patch_function=pipeline.init,
        manage_run=False,
    )

    # TODO: Check folder Pipeline to Artifact to getML
    # TODO: Log metadata on Dataset
    # TODO: Add project/pipeline path option to autologging

    safe_patch(
        autologging_integration=FLAVOR_NAME,
        destination=getml.pipeline.Pipeline,
        function_name="fit",
        patch_function=with_mlflowclient(mlflowclient)(pipeline.fit),
        manage_run=False,
    )

    safe_patch(
        autologging_integration=FLAVOR_NAME,
        destination=getml.pipeline.Pipeline,
        function_name="score",
        patch_function=with_mlflowclient(mlflowclient)(pipeline.score),
        manage_run=False,
    )

    for destination in (getml, getml.engine, getml.engine.helpers):
        safe_patch(
            autologging_integration=FLAVOR_NAME,
            destination=destination,
            function_name="set_project",
            patch_function=with_mlflowclient(mlflowclient)(engine.set_project),
            manage_run=False,
        )

    safe_patch(
        autologging_integration=FLAVOR_NAME,
        destination=getml.pipeline.Pipeline,
        function_name="predict",
        patch_function=with_mlflowclient(mlflowclient)(pipeline.predict),
        manage_run=False,
    )
