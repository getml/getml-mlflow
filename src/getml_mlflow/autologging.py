from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Optional

import functools

import getml
import mlflow
from mlflow import MlflowClient
from mlflow.utils.autologging_utils import autologging_integration
from mlflow.utils.autologging_utils.safety import revert_patches, safe_patch

import getml_mlflow.logging.logger
from getml_mlflow.flavor import FLAVOR_NAME
from getml_mlflow.loggingconfiguration import LoggingConfiguration
from getml_mlflow.patch import engine, pipeline, project


def with_logging_configuration(
    logging_configuration: LoggingConfiguration,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return func(*args, **kwargs, logging_configuration=logging_configuration)

        return wrapper

    return decorator


DEFAULT_MLFLOW_TRACKING_URI = "http://localhost:5000"


@autologging_integration(FLAVOR_NAME)
def autolog(
    *,
    log_data_container_information: bool = True,
    log_data_container_as_artifact: bool = True,
    log_function_parameters: bool = True,
    log_function_return: bool = True,
    log_pipeline_parameters: bool = True,
    log_pipeline_tags: bool = True,
    log_pipeline_scores: bool = True,
    log_pipeline_features: bool = True,
    log_pipeline_columns: bool = True,
    log_pipeline_targets: bool = True,
    log_pipelne_as_artifact: bool = True,
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

    getml_mlflow.logging.logger.set_up()
    tracking_uri = tracking_uri or DEFAULT_MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)

    logging_configuration: LoggingConfiguration = LoggingConfiguration(
        mlflow_client=MlflowClient(tracking_uri=tracking_uri),
        log_data_container_information=log_data_container_information,
        log_data_container_as_artifact=log_data_container_as_artifact,
        log_function_parameters=log_function_parameters,
        log_function_return=log_function_return,
        log_pipeline_parameters=log_pipeline_parameters,
        log_pipeline_tags=log_pipeline_tags,
        log_pipeline_scores=log_pipeline_scores,
        log_pipeline_features=log_pipeline_features,
        log_pipeline_columns=log_pipeline_columns,
        log_pipeline_targets=log_pipeline_targets,
        log_pipeline_as_artifact=log_pipelne_as_artifact,
        log_system_metrics=log_system_metrics,
        silent=silent,
        create_runs=create_runs,
        extra_tags=extra_tags,
        getml_project_path=getml_project_path,
    )

    for destination in (getml, getml.engine, getml.engine.helpers):
        safe_patch(
            autologging_integration=FLAVOR_NAME,
            destination=destination,
            function_name="set_project",
            patch_function=with_logging_configuration(logging_configuration)(
                engine.set_project
            ),
            manage_run=False,
        )

    safe_patch(
        autologging_integration=FLAVOR_NAME,
        destination=getml.project.attrs,
        function_name="switch",
        patch_function=with_logging_configuration(logging_configuration)(
            project.switch
        ),
        manage_run=False,
    )

    safe_patch(
        autologging_integration=FLAVOR_NAME,
        destination=getml.pipeline.Pipeline,
        function_name="__init__",
        patch_function=pipeline.init,
        manage_run=False,
    )

    for destination in (getml.pipeline, getml.pipeline.helpers2):
        safe_patch(
            autologging_integration=FLAVOR_NAME,
            destination=destination,
            function_name="load",
            patch_function=with_logging_configuration(logging_configuration)(
                pipeline.load
            ),
            manage_run=False,
        )

    # TODO: Check folder Pipeline to Artifact to getML
    # TODO: log data model -> SVG
    # TODO: add current user as name
    # TODO: check copilot agent for docstring generation

    safe_patch(
        autologging_integration=FLAVOR_NAME,
        destination=getml.pipeline.Pipeline,
        function_name="fit",
        patch_function=with_logging_configuration(logging_configuration)(pipeline.fit),
        manage_run=False,
    )

    safe_patch(
        autologging_integration=FLAVOR_NAME,
        destination=getml.pipeline.Pipeline,
        function_name="score",
        patch_function=with_logging_configuration(logging_configuration)(
            pipeline.score
        ),
        manage_run=False,
    )

    safe_patch(
        autologging_integration=FLAVOR_NAME,
        destination=getml.pipeline.Pipeline,
        function_name="predict",
        patch_function=with_logging_configuration(logging_configuration)(
            pipeline.predict
        ),
        manage_run=False,
    )

    safe_patch(
        autologging_integration=FLAVOR_NAME,
        destination=getml.pipeline.Pipeline,
        function_name="transform",
        patch_function=with_logging_configuration(logging_configuration)(
            pipeline.transform
        ),
        manage_run=False,
    )
