from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple

import getml
from getml.pipeline import Pipeline
from mlflow.client import MlflowClient

from getml_mlflow.constants import DEFAULT_GETML_PROJECTS_PATH


def log_pipeline_as_artifact(
    mlflow_client: MlflowClient,
    run_id: str,
    pipeline: Pipeline,
    *,
    project_name: Optional[str] = None,
    projects_path: Path = DEFAULT_GETML_PROJECTS_PATH,
) -> str:
    if project_name is None:
        project_name = getml.project.name

    pipeline._save()

    artifact_path: str = f"pipeline/{project_name}"

    for item in (projects_path / project_name).iterdir():
        if item.is_dir():
            if item.stem == "data":
                continue
            if item.stem == "pipelines":
                mlflow_client.log_artifact(
                    run_id,
                    item / pipeline.id,
                    f"{artifact_path}/pipelines",
                )
                continue

        mlflow_client.log_artifact(run_id, item, artifact_path)

    return pipeline.id


def download_artifact_pipeline(
    mlflow_client: MlflowClient,
    run_id: str,
    pipeline_id: str,
    *,
    original_project_name: Optional[str] = None,
    projects_path: Path = DEFAULT_GETML_PROJECTS_PATH,
) -> Tuple[str, str]:
    """
    Downloads a getML pipeline artifact from an MLflow run and saves it as a new project.

    This function downloads a pipeline artifact from MLflow and creates a new project with
    a name derived from the original project name and the pipeline ID: 
    "original_project_name-pipeline_id".

    If the project already exists (e.g., when calling this function multiple times with 
    the same parameters), the existing project will be overwritten with the downloaded 
    artifacts.

    ??? warning "Experimental feature"
        This feature is experimental and may change in future releases.

    Args:
        mlflow_client: An MLflow client instance to interact with MLflow.

        run_id: The ID of the MLflow run containing the pipeline artifacts.

        pipeline_id: The ID of the pipeline to be downloaded.

        original_project_name: The name of the original getML project the pipeline was 
            saved from. If None, uses the current project name.

        projects_path: Path where getML projects are stored. Defaults to 
            `$HOME/.getML/projects`.

    Returns:
        A tuple containing:
            - The name of the newly created getML project
            - The ID of the downloaded pipeline

    Example:
        ```python
        # Initialize MLflow client
        client = MlflowClient(tracking_uri="http://localhost:5000")

        run_id = "abcdef1234567890"
        pipeline_id = "l2TCiD"

        # Download pipeline artifact from a specific run. This creates a new project 
        # named "interstate94-l2TCiD" with the pipeline
        new_project, pipeline_id = getml_mlflow.marshalling.pipeline.download_artifact_pipeline(
            client, run_id, pipeline_id, original_project_name="interstate94"
            )

        # You can now switch to the new project and load the pipeline
        getml.project.set_project(new_project)
        pipeline = getml.pipeline.load(pipeline_id)
        ```

    """
    
    if original_project_name is None:
        original_project_name = getml.project.name

    with TemporaryDirectory() as temp_dir:
        mlflow_client.download_artifacts(
            run_id, f"pipeline/{original_project_name}", temp_dir
        )
        new_project_name: str = f"{original_project_name}-{pipeline_id}"
        project_path: Path = projects_path / new_project_name
        temp_project_path: Path = Path(temp_dir) / "pipeline" / original_project_name
        temp_project_path.rename(project_path)

    return (new_project_name, pipeline_id)


def switch_to_artifact_pipeline(
    mlflow_client: MlflowClient,
    run_id: str,
    pipeline_id: str,
    *,
    original_project_name: Optional[str] = None,
    projects_path: Path = DEFAULT_GETML_PROJECTS_PATH,
) -> Pipeline:
    """
    Downloads an artifact pipeline from MLflow, switches to the newly created project,
    and loads the pipeline.

    This function simplifies the workflow of retrieving a pipeline stored as an MLflow 
    artifact. It downloads the pipeline into a new getML project (named as 
    "original_project_name-pipeline_id"), automatically switches to that project, and 
    loads the pipeline for immediate use.

    ??? warning "Experimental feature"
        This function is experimental and may change in future releases.

    Args:
        mlflow_client: The MLflow client instance to use for retrieving the artifact.

        run_id: The ID of the MLflow run containing the pipeline artifact.

        pipeline_id: The ID of the pipeline to download.

        original_project_name: The name of the original project. If None, the current 
            project name is used. Defaults to None.

        projects_path: Path to the getML projects directory. Defaults to 
            `$HOME/.getML/projects`.

    Returns:
        Pipeline: The loaded pipeline object in the newly created project.

    Examples:
        ```python
        import mlflow
        from mlflow.tracking import MlflowClient
        import getml_mlflow
        import getml
        
        # Connect to MLflow
        client = MlflowClient("http://localhost:5000")
        
        # Download pipeline from run and switch to new project
        pipeline = getml_mlflow.marshalling.pipeline.switch_to_artifact_pipeline(
                   client,
                   "2960ee40202744daa64aa83d180f0b2f",
                   "uPe3hR"
                )
        
        # Pipeline is ready to use
        predictions = pipeline.predict(container.test)
        ```

    """
    if original_project_name is None:
        original_project_name = getml.project.name

    project_name, pipeline_id = download_artifact_pipeline(
        mlflow_client=mlflow_client,
        run_id=run_id,
        pipeline_id=pipeline_id,
        original_project_name=original_project_name,
        projects_path=projects_path,
    )
    getml.project.switch(project_name)
    return getml.pipeline.load(pipeline_id)
