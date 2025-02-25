from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, List, Optional

import functools

import getml
import mlflow
from mlflow import MlflowClient
from mlflow.utils.autologging_utils import autologging_integration
from mlflow.utils.autologging_utils.safety import revert_patches, safe_patch

from getml_mlflow.constants import DEFAULT_MLFLOW_TRACKING_URI
from getml_mlflow.flavor import FLAVOR_NAME
from getml_mlflow.loggingconfiguration import LoggingConfiguration
from getml_mlflow.patch import engine, pipeline, project
from getml_mlflow.util.with_kwargs import with_kwargs


@dataclass
class SafePatchFunction:
    destination: Any
    function_name: str
    patch_function: Any
    with_logging_configuration: bool = True


FUNCTIONS_TO_PATCH: List[SafePatchFunction] = [
    SafePatchFunction(
        destination=getml,
        function_name="set_project",
        patch_function=engine.set_project,
    ),
    SafePatchFunction(
        destination=getml.engine,
        function_name="set_project",
        patch_function=engine.set_project,
    ),
    SafePatchFunction(
        destination=getml.engine.helpers,
        function_name="set_project",
        patch_function=engine.set_project,
    ),
    SafePatchFunction(
        destination=getml.project.attrs,
        function_name="switch",
        patch_function=project.switch,
    ),
    SafePatchFunction(
        destination=getml.pipeline.Pipeline,
        function_name="__init__",
        patch_function=pipeline.init,
        with_logging_configuration=False,
    ),
    SafePatchFunction(
        destination=getml.pipeline,
        function_name="load",
        patch_function=pipeline.load,
    ),
    SafePatchFunction(
        destination=getml.pipeline.helpers2,
        function_name="load",
        patch_function=pipeline.load,
    ),
    SafePatchFunction(
        destination=getml.pipeline.Pipeline,
        function_name="fit",
        patch_function=pipeline.fit,
    ),
    SafePatchFunction(
        destination=getml.pipeline.Pipeline,
        function_name="score",
        patch_function=pipeline.score,
    ),
    SafePatchFunction(
        destination=getml.pipeline.Pipeline,
        function_name="predict",
        patch_function=pipeline.predict,
    ),
    SafePatchFunction(
        destination=getml.pipeline.Pipeline,
        function_name="transform",
        patch_function=pipeline.transform,
    ),
]


def with_logging_configuration(
    logging_configuration: LoggingConfiguration,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return func(*args, **kwargs, logging_configuration=logging_configuration)

        return wrapper

    return decorator


@autologging_integration(FLAVOR_NAME)
def autolog(
    *,
    log_data_container_information: bool = True,
    log_data_container_as_artifact: bool = True,
    log_function_parameters: bool = True,
    log_function_return: bool = True,
    log_function_as_trace: bool = True,
    log_pipeline_parameters: bool = True,
    log_pipeline_tags: bool = True,
    log_pipeline_scores: bool = True,
    log_pipeline_features: bool = True,
    log_pipeline_columns: bool = True,
    log_pipeline_targets: bool = True,
    log_pipeline_data_model: bool = True,
    log_pipeline_as_artifact: bool = True,
    log_system_metrics: bool = True,
    disable: bool = False,
    silent: bool = False,
    create_runs: bool = True,
    extra_tags: Optional[Dict[str, str]] = None,
    getml_project_path: Optional[str] = None,
    tracking_uri: Optional[str] = None,
) -> None:
    if disable:
        revert_patches(FLAVOR_NAME)
        return

    tracking_uri = tracking_uri or DEFAULT_MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)

    logging_configuration: LoggingConfiguration = LoggingConfiguration(
        mlflow_client=MlflowClient(tracking_uri=tracking_uri),
        data_container=LoggingConfiguration.DataContainer(
            log_information=log_data_container_information,
            log_as_artifact=log_data_container_as_artifact,
        ),
        function=LoggingConfiguration.Function(
            log_parameters=log_function_parameters,
            log_return=log_function_return,
            log_as_trace=log_function_as_trace,
        ),
        pipeline=LoggingConfiguration.Pipeline(
            log_parameters=log_pipeline_parameters,
            log_tags=log_pipeline_tags,
            log_scores=log_pipeline_scores,
            log_features=log_pipeline_features,
            log_columns=log_pipeline_columns,
            log_targets=log_pipeline_targets,
            log_data_model=log_pipeline_data_model,
            log_as_artifact=log_pipeline_as_artifact,
        ),
        log_system_metrics=log_system_metrics,
        silent=silent,
        create_runs=create_runs,
        extra_tags=extra_tags,
        getml_project_path=getml_project_path,
    )

    for function in FUNCTIONS_TO_PATCH:
        safe_patch(
            autologging_integration=FLAVOR_NAME,
            destination=function.destination,
            function_name=function.function_name,
            patch_function=(
                with_kwargs(logging_configuration=logging_configuration)(
                    function.patch_function
                )
                if function.with_logging_configuration
                else function.patch_function
            ),
            manage_run=False,
        )
