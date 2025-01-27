from tempfile import NamedTemporaryFile

import mlflow
import numpy
from numpy.typing import NDArray


def log_ndarray_as_artifact(data: NDArray[numpy.float_], artifact_path: str):
    with NamedTemporaryFile(suffix=".npy") as temp_file:
        numpy.save(temp_file, data)
        mlflow.log_artifact(temp_file.name, artifact_path)
