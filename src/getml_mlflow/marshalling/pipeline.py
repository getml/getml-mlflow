import pathlib
import shutil
from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple, Union
import getml
import mlflow
import mlflow.client
from mlflow.utils.server_cli_utils import artifacts_only_config_validation


class Pipeline:
    def __init__(
        self,
        mlflowclient: mlflow.client.MlflowClient,
        run_id: str,
        projects_path: pathlib.Path = pathlib.Path.home() / ".getML" / "projects",
        project_name: str = getml.project.name,
    ) -> None:
        self._mlflowclient: mlflow.client.MlflowClient = mlflowclient
        self._run_id: str = run_id
        self._project_name: str = project_name
        self._projects_path: pathlib.Path = projects_path
        self._project_path: pathlib.Path = projects_path / project_name

    def serialize(
        self,
        pipeline: getml.pipeline.Pipeline,
    ) -> str:
        pipeline._save()

        artifact_path: str = f"pipeline/{self._project_name}"

        for item in self._project_path.iterdir():
            if item.is_dir():
                if item.stem == "data":
                    continue
                if item.stem == "pipelines":
                    self._mlflowclient.log_artifact(
                        self._run_id,
                        item / pipeline.id,
                        f"{artifact_path}/pipelines",
                    )
                    continue

            self._mlflowclient.log_artifact(self._run_id, item, f"{artifact_path}")

        return pipeline.id

    def deserialize(self, id: str) -> Tuple[str, str]:
        with TemporaryDirectory() as temp_dir:
            self._mlflowclient.download_artifacts(
                self._run_id, f"pipeline/{self._project_name}", temp_dir
            )
            project_name: str = f"{self._project_name}-{id}"
            project_path: pathlib.Path = self._projects_path / project_name
            temp_project_path: pathlib.Path = (
                pathlib.Path(temp_dir) / "pipeline" / self._project_name
            )
            shutil.move(temp_project_path, project_path)

        return (project_name, id)
        # getml.project.switch(project_name)
        # return getml.pipeline.load(id)

    @staticmethod
    def load(project_name: str, pipeline_id: str) -> getml.Pipeline:
        getml.project.switch(project_name)
        return getml.pipeline.load(pipeline_id)
