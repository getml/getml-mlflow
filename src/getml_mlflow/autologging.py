from typing import Dict, Optional

import getml
from mlflow.utils.autologging_utils import autologging_integration
from mlflow.utils.autologging_utils.safety import safe_patch

from getml_mlflow.flavor import FLAVOR_NAME
from getml_mlflow.patch import engine, pipeline


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
    safe_patch(
        autologging_integration=FLAVOR_NAME,
        destination=getml.pipeline.Pipeline,
        function_name="__init__",
        patch_function=pipeline.init,
        manage_run=False,
    )

    # TODO: Log Pipeline on fit, because of new ID
    # TODO: Add Pipeline-ID to RUN Name
    # TODO: Check folder Pipeline to Artifact to getML
    # TODO: Log metadata on Dataset
    # TODO: Add project/pipeline path option to autologging

    safe_patch(
        autologging_integration=FLAVOR_NAME,
        destination=getml.pipeline.Pipeline,
        function_name="fit",
        patch_function=pipeline.fit,
        manage_run=False,
    )

    safe_patch(
        autologging_integration=FLAVOR_NAME,
        destination=getml.pipeline.Pipeline,
        function_name="score",
        patch_function=pipeline.score,
        manage_run=False,
    )

    for destination in (getml, getml.engine, getml.engine.helpers):
        safe_patch(
            autologging_integration=FLAVOR_NAME,
            destination=destination,
            function_name="set_project",
            patch_function=engine.set_project,
            manage_run=False,
        )

    safe_patch(
        autologging_integration=FLAVOR_NAME,
        destination=getml.pipeline.Pipeline,
        function_name="predict",
        patch_function=pipeline.predict,
        manage_run=False,
    )
