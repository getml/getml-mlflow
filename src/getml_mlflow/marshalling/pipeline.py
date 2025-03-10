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
