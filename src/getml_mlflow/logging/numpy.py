from tempfile import TemporaryDirectory
from typing import Optional

import mlflow
import numpy
from numpy.typing import NDArray


class NumpyLogger:
    def __init__(
        self,
        mlflow_client: mlflow.MlflowClient,
        run_id: str,
    ) -> None:
        self._mlflow_client: mlflow.MlflowClient = mlflow_client
        self._run_id: str = run_id

    def log_ndarray_as_artifact(
        self,
        data: NDArray[numpy.float_],
        name: str,
        artifact_path: Optional[str] = None,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            filename: str = f"{name}.npy"
            local_path: str = f"{temp_dir}/{filename}"
            numpy.save(local_path, data)
            self._mlflow_client.log_artifact(
                run_id=self._run_id,
                local_path=local_path,
                artifact_path=artifact_path,
            )
