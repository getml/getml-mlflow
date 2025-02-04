from enum import StrEnum
from tempfile import TemporaryDirectory
from typing import Callable, Dict, List, Sequence, Union

import mlflow
import mlflow.data.code_dataset_source
import mlflow.data.pandas_dataset
from mlflow.entities import DatasetInput, InputTag
from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT

import getml
from getml_mlflow.data import dataframelike, getml_dataset
from getml_mlflow.data.dataframelike import DataFrameLike


class DataContainerLoggerTarget(StrEnum):
    ARTIFACT = "artifact"
    INPUT = "input"


class DataContainerLogger:
    @classmethod
    def as_artifact(cls, mlflowclient: mlflow.MlflowClient, run_id: str):
        return cls(mlflowclient, run_id, DataContainerLoggerTarget.ARTIFACT)

    @classmethod
    def as_input(cls, mlflowclient: mlflow.MlflowClient, run_id: str):
        return cls(mlflowclient, run_id, DataContainerLoggerTarget.INPUT)

    def __init__(
        self,
        mlflowclient: mlflow.MlflowClient,
        run_id: str,
        target: DataContainerLoggerTarget,
    ):
        self._mlflowclient: mlflow.MlflowClient = mlflowclient
        self._run_id: str = run_id
        self._run = mlflow.get_run(run_id)
        self._seperator: Callable[[], str] = self._get_seperator(target)
        self._log_dataframe_like: Callable[[DataFrameLike, List[str]], None] = (
            self._get_log_dataframe_like(target)
        )

    def _get_seperator(self, target: DataContainerLoggerTarget):
        if target == DataContainerLoggerTarget.ARTIFACT:
            return self._artifact_path_seperator
        elif target == DataContainerLoggerTarget.INPUT:
            return self._input_context_seperator
        else:
            raise ValueError(f"Unknown target: {target}")

    def _get_log_dataframe_like(self, target: DataContainerLoggerTarget):
        if target == DataContainerLoggerTarget.ARTIFACT:
            return self._log_dataframe_like_as_artifact
        elif target == DataContainerLoggerTarget.INPUT:
            return self._log_dataframe_like_as_input
        else:
            raise ValueError(f"Unknown target: {target}")

    def log_data_containers(
        self,
        data_containers: Union[
            Sequence[DataFrameLike],
            Dict[str, DataFrameLike],
        ],
        prefix: str,
    ) -> None:
        if isinstance(data_containers, dict):
            for name, data_container in data_containers.items():
                self.log_data_container(data_container, [prefix, name])
        else:
            for id, data_container in enumerate(data_containers):
                self.log_data_container(data_container, [prefix, str(id)])

    def log_data_container(
        self,
        data_container: Union[DataFrameLike, getml.data.Subset],
        context: Union[str, List[str]],
    ) -> None:
        if isinstance(context, str):
            context = [context]
        if isinstance(data_container, (getml.DataFrame, getml.data.View)):
            self._log_dataframe_like(data_container, context)
        elif isinstance(data_container, getml.data.Subset):
            self._log_subset(data_container, context)

    def _log_dataframe_like_as_input(
        self, dataframe_like: DataFrameLike, context: List[str]
    ) -> None:
        dataset: getml_dataset.GetMLDataset = getml_dataset.GetMLDataset(dataframe_like)

        dataset_context: str = self._seperator().join(context)
        dataset_input: DatasetInput = DatasetInput(
            dataset=dataset._to_mlflow_entity(),
            tags=[InputTag(key=MLFLOW_DATASET_CONTEXT, value=dataset_context)],
        )
        self._mlflowclient.log_inputs(
            run_id=self._run_id,
            datasets=[dataset_input],
        )
        self._mlflowclient.log_dict(
            self._run_id,
            dataset.as_dict(),
            f"input/dataset/{dataset_context}.{dataset.name}.json",
        )

    def _log_dataframe_like_as_artifact(
        self, dataframe_like: DataFrameLike, context: List[str]
    ) -> None:
        filename: str = dataframelike.get_name(dataframe_like) + ".parquet"
        artifact_path: str = self._seperator().join(context)
        with TemporaryDirectory() as temp_dir:
            local_path: str = f"{temp_dir}/{filename}"
            dataframe_like.to_parquet(local_path)
            artifact_path: str = self._seperator().join(context)
            self._mlflowclient.log_artifact(
                run_id=self._run_id,
                local_path=local_path,
                artifact_path=artifact_path,
            )

        dataset: getml_dataset.GetMLDataset = getml_dataset.GetMLDataset(dataframe_like)
        self._mlflowclient.log_inputs(
            run_id=self._run_id,
            datasets=[DatasetInput(dataset=dataset._to_mlflow_entity())],
        )

        self._mlflowclient.log_dict(
            self._run_id, dataset.as_dict(), f"{artifact_path}/{dataset.name}.json"
        )

    def _log_subset(self, subset: getml.data.Subset, context: List[str]) -> None:
        self._log_dataframe_like(subset.population, context + ["Population"])

        for name, table in subset.peripheral.items():
            self._log_dataframe_like(table, context + ["Peripheral", name])

    def _artifact_path_seperator(self) -> str:
        return "/"

    def _input_context_seperator(self) -> str:
        return "."
