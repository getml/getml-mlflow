from tempfile import NamedTemporaryFile

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
        artifact_path: str,
    ):
        with NamedTemporaryFile(suffix=".npy") as temp_file:
            numpy.save(temp_file, data)
            self._mlflowclient.log_artifact(
                run_id=self._run_id,
                local_path=temp_file.name,
                artifact_path=artifact_path,
            )
