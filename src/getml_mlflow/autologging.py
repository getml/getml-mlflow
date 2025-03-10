from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import getml
import mlflow
from mlflow import MlflowClient
from mlflow.utils.autologging_utils import autologging_integration
from mlflow.utils.autologging_utils.safety import revert_patches, safe_patch

from getml_mlflow.constants import DEFAULT_MLFLOW_TRACKING_URI
from getml_mlflow.flavor import FLAVOR_NAME
from getml_mlflow.loggingconfiguration import (
    DataContainerLoggingConfiguration,
    FunctionLoggingConfiguration,
    GeneralLoggingConfiguration,
    LoggingConfiguration,
    PipelineLoggingConfiguration,
)
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


@autologging_integration(FLAVOR_NAME)
def autolog(
    *,
    log_data_information: bool = True,
    log_data_as_artifact: bool = True,
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
    getml_project_path: Optional[Path] = None,
    tracking_uri: Optional[str] = None,
) -> None:
    """Enable automatic logging of getML operations to MLflow.

    This function enables automatic logging of the following operations to MLflow:

    - pipeline creation, loading and operations (fit, score, predict, transform)
    - project setting and switching.

    Pipeline parameters, performance metrics, dataframe metadata, and other relevant
    information are captured and displayed in the MLflow UI. It also allows logging of
    dataframes passed as a function parameter or returned by a function as artifacts.

    The artifacts are stored in `artifacts-destination` set when running
    `mlflow ui` command. The default is `artifacts` directory in the current
    working directory.

    In the UI, getML pipelines correspond to MLflow runs, functions correspond to
    sub-runs, and projects correspond to experiments.

    For a detailed introduction on this MLflow integration, including setup, working with
    artifact pipelines, and more, please refer to our
    [Tracking with MLflow][mlflow-integration-guide] guide. The guide provides examples
    and configuration options to help you get the most from it.

    Args:
        log_data_information (bool, optional): Whether to log metadata about
            a `Dataframe` or `View` (e.g., number of rows & columns, column names, roles).

            The [`roles`][getml.data.roles] are indicated with the
            following emojis in the MLflow UI:

            - 🗃 for categorical columns
            - 🔗 for join keys
            - 🔢 for numerical columns
            - 🎯 for target column(s)
            - 📝 for text columns
            - ⏰ for timestamp columns
            - 🧮 for unused float columns
            - 🧵 for unused string columns

        log_data_as_artifact (bool, optional): Whether to log a `DataFrame`,
            `View` or `Subset` function parameter as a `.parquet` artifact. In addition,
            it allows logging of `DataFrame` returned by functions. In MLflow UI, the
            artifacts are available to download. `log_function_parameters` or
            `log_function_return` must be `True` for this to work.

        log_function_parameters (bool, optional): Whether to log parameters passed to
            getML functions, e.g., pipe.fit() in the MLflow UI. To log the `DataFrame`,
            `View` or `Subset` function parameters as artifacts,
            `log_data_as_artifact=True` must also be set to `True`.

        log_function_return (bool, optional): Whether to log return values of getML
            functions as artifacts. For example, it enables logging of `DataFrame`
            (as `.parquet`) and `numpy.ndarray` (as `.npy`) returned by `transform()`
            or `predict()` methods. `log_data_as_artifact` must also be `True`
            for `DataFrame` logging.

        log_function_as_trace (bool, optional): Whether to log function calls as MLflow
            traces for detailed execution flow.

        log_pipeline_parameters (bool, optional): Whether to log
            [`parameters`][getml.pipeline.Pipeline] of a pipeline.

        log_pipeline_tags (bool, optional): Whether to log
            [`tags`][getml.pipeline.Pipeline] of a pipeline.

        log_pipeline_scores (bool, optional): Whether to log [`scores`][getml.pipeline.Scores]
            (metrics) of a pipeline.

        log_pipeline_features (bool, optional): Whether to log [`features`][getml.pipeline.Features]
            learned during pipeline fitting.

        log_pipeline_columns (bool, optional): Whether to log [`columns`][getml.pipeline.Columns]
            (whose importance can be calculated) of a pipeline.

        log_pipeline_targets (bool, optional): Whether to log
            [`targets`][getml.pipeline.Pipeline.targets] of a pipeline.

        log_pipeline_data_model (bool, optional): Whether to log the
            [`data model`][getml.data.DataModel] provided in the pipeline. It is available
            as an HTML artifact to view or download.

        log_pipeline_as_artifact (bool, optional): Whether to save pipelines as
            MLflow artifacts.

            ??? note "Docker configuration, `download_artifact_pipeline()` and `switch_to_artifact_pipeline()`"

                When using this parameter with Docker, you'll need to set up proper bind
                mounts to allow pipeline artifact logging. For detailed instructions on
                working with artifact pipelines, Docker configurations, and related
                functions like `download_artifact_pipeline()` and `switch_to_artifact_pipeline()`,
                please refer to the [Tracking with MLflow][mlflow-integration-guide] guide.

        log_system_metrics (bool, optional): Whether to log system metrics (CPU, memory usage)
            during pipeline fitting. Metrics are available for getML Enterprise only.

        disable (bool, optional): If True, disables all getML autologging.

        silent (bool, optional): If True, suppresses all informational logging messages.

        create_runs (bool, optional): If True, creates new MLflow runs automatically
            when logging. You may set it to False and log under your own run. For example:

            ```python
            import mlflow
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment("your_experiment_name")
            with mlflow.start_run(run_name="your_run_name"):
                pipe.fit(container.train)
            ```

        extra_tags (Dict[str, str], optional): Additional custom tags to log with each MLflow run.

        getml_project_path (Path, optional): Path to the getML projects directory.
            Used for accessing and logging pipeline artifacts when
            `log_pipeline_as_artifact=True`. If not provided, defaults to
            `$HOME/.getML/projects`.

        tracking_uri (str, optional): MLflow tracking server URI. If not provided,
            uses `http://localhost:5000`.


    Examples:
        Basic usage with default settings:

        ```python
        import getml
        import getml_mlflow
        getml_mlflow.autolog()
        # Subsequent getML pipeline operations will be logged to MLflow
        ```

        Custom configuration:
        ```python
        getml_mlflow.autolog(
            log_pipeline_as_artifact=True,
            log_system_metrics=False,
            tracking_uri="http://localhost:5000"
        )
        ```
    """
    if disable:
        revert_patches(FLAVOR_NAME)
        return

    tracking_uri = tracking_uri or DEFAULT_MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)

    logging_configuration: LoggingConfiguration = LoggingConfiguration(
        general=GeneralLoggingConfiguration(
            mlflow_client=MlflowClient(tracking_uri=tracking_uri),
            log_system_metrics=log_system_metrics,
            silent=silent,
            create_runs=create_runs,
            extra_tags=extra_tags,
            getml_project_path=getml_project_path,
        ),
        data_container=DataContainerLoggingConfiguration(
            log_information=log_data_information,
            log_as_artifact=log_data_as_artifact,
        ),
        function=FunctionLoggingConfiguration(
            log_parameters=log_function_parameters,
            log_return=log_function_return,
            log_as_trace=log_function_as_trace,
        ),
        pipeline=PipelineLoggingConfiguration(
            log_parameters=log_pipeline_parameters,
            log_tags=log_pipeline_tags,
            log_scores=log_pipeline_scores,
            log_features=log_pipeline_features,
            log_columns=log_pipeline_columns,
            log_targets=log_pipeline_targets,
            log_data_model=log_pipeline_data_model,
            log_as_artifact=log_pipeline_as_artifact,
        ),
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
