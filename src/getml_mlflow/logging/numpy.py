from tempfile import TemporaryDirectory

import mlflow
import numpy
from numpy.typing import NDArray


class NumpyLogger:
    def __init__(self, mlflowclient: mlflow.MlflowClient, run_id: str) -> None:
        self._mlflowclient: mlflow.MlflowClient = mlflowclient
        self._run_id: str = run_id

    def log_ndarray_as_artifact(
        self,
        data: NDArray[numpy.float_],
        name: str,
        artifact_path: str = "",
    ):
        with TemporaryDirectory() as temp_dir:
            filename = f"{name}.npy"
            local_path: str = f"{temp_dir}/{filename}"
            numpy.save(local_path, data)
            self._mlflowclient.log_artifact(
                run_id=self._run_id,
                local_path=local_path,
                artifact_path=artifact_path,
            )
