import pathlib
import shutil
from tempfile import TemporaryDirectory
from typing import Optional, Tuple

import getml
import mlflow
import mlflow.client

from getml_mlflow.constants import DEFAULT_GETML_PROJECTS_PATH


def log_pipeline_as_artifact(
    mlflowclient: mlflow.client.MlflowClient,
    run_id: str,
    pipeline: getml.pipeline.Pipeline,
    *,
    project_name: Optional[str] = None,
    projects_path: pathlib.Path = DEFAULT_GETML_PROJECTS_PATH,
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
                mlflowclient.log_artifact(
                    run_id,
                    item / pipeline.id,
                    f"{artifact_path}/pipelines",
                )
                continue

        mlflowclient.log_artifact(run_id, item, f"{artifact_path}")

    return pipeline.id


def download_artifact_pipeline(
    mlflowclient: mlflow.client.MlflowClient,
    run_id: str,
    pipeline_id: str,
    *,
    original_project_name: Optional[str] = None,
    projects_path: pathlib.Path = DEFAULT_GETML_PROJECTS_PATH,
) -> Tuple[str, str]:
    if original_project_name is None:
        original_project_name = getml.project.name

    with TemporaryDirectory() as temp_dir:
        mlflowclient.download_artifacts(
            run_id, f"pipeline/{original_project_name}", temp_dir
        )
        new_project_name: str = f"{original_project_name}-{pipeline_id}"
        project_path: pathlib.Path = projects_path / new_project_name
        temp_project_path: pathlib.Path = (
            pathlib.Path(temp_dir) / "pipeline" / original_project_name
        )
        shutil.move(temp_project_path, project_path)

    return (new_project_name, pipeline_id)


def switch_to_artifact_pipeline(
    mlflowclient: mlflow.client.MlflowClient,
    run_id: str,
    pipeline_id: str,
    *,
    original_project_name: Optional[str] = None,
    projects_path: pathlib.Path = DEFAULT_GETML_PROJECTS_PATH,
) -> getml.pipeline.Pipeline:
    if original_project_name is None:
        original_project_name = getml.project.name

    project_name, pipeline_id = download_artifact_pipeline(
        mlflowclient=mlflowclient,
        run_id=run_id,
        pipeline_id=pipeline_id,
        original_project_name=original_project_name,
        projects_path=projects_path,
    )
    getml.project.switch(project_name)
    return getml.pipeline.load(pipeline_id)
