from __future__ import annotations

from enum import StrEnum
from functools import cached_property
from tempfile import TemporaryDirectory
from typing import Dict, List, Sequence, Union

import getml
import mlflow
import mlflow.data.code_dataset_source
import mlflow.data.pandas_dataset
import mlflow.entities
from mlflow.entities import DatasetInput, InputTag
from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT

from getml_mlflow.data import dataframelike, getml_dataset
from getml_mlflow.data.dataframelike import DataFrameLike


class DataContainerLoggerTarget(StrEnum):
    ARTIFACT = "artifact"
    INPUT = "input"


class DataContainerLogger:
    @classmethod
    def as_artifact(
        cls,
        mlflowclient: mlflow.MlflowClient,
        run_id: str,
        *,
        log_information: bool = True,
        log_as_artifact: bool = True,
    ) -> DataContainerLogger:
        return cls(
            mlflowclient,
            run_id,
            DataContainerLoggerTarget.ARTIFACT,
            log_information=log_information,
            log_as_artifact=log_as_artifact,
        )

    @classmethod
    def as_input(
        cls,
        mlflowclient: mlflow.MlflowClient,
        run_id: str,
        *,
        log_information: bool = True,
        log_as_artifact: bool = True,
    ) -> DataContainerLogger:
        return cls(
            mlflowclient,
            run_id,
            DataContainerLoggerTarget.INPUT,
            log_information=log_information,
            log_as_artifact=log_as_artifact,
        )

    def __init__(
        self,
        mlflowclient: mlflow.MlflowClient,
        run_id: str,
        target: DataContainerLoggerTarget,
        *,
        log_information: bool = True,
        log_as_artifact: bool = True,
    ) -> None:
        self._mlflowclient: mlflow.MlflowClient = mlflowclient
        self._run_id: str = run_id
        self._run: mlflow.entities.Run = self._mlflowclient.get_run(run_id)
        self._target: DataContainerLoggerTarget = target
        self._log_information: bool = log_information
        self._log_as_artifact: bool = log_as_artifact

    @cached_property
    def _separator(self) -> str:
        if self._target == DataContainerLoggerTarget.ARTIFACT:
            return "/"
        elif self._target == DataContainerLoggerTarget.INPUT:
            return "."
        else:
            raise ValueError(f"Unknown target: {self._target}")

    def _log_dataframe_like(
        self, dataframe_like: DataFrameLike, context: List[str]
    ) -> None:
        if self._target == DataContainerLoggerTarget.ARTIFACT:
            return self._log_dataframe_like_as_artifact(dataframe_like, context)
        elif self._target == DataContainerLoggerTarget.INPUT:
            return self._log_dataframe_like_as_input(dataframe_like, context)
        else:
            raise ValueError(f"Unknown target: {self._target}")

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
        if isinstance(data_container, (getml.data.DataFrame, getml.data.View)):
            self._log_dataframe_like(data_container, context)
        elif isinstance(data_container, getml.data.Subset):
            self._log_subset(data_container, context)

    def _log_dataframe_like_as_input(
        self, dataframe_like: DataFrameLike, context: List[str]
    ) -> None:
        if not self._log_information:
            return

        dataset: getml_dataset.GetMLDataset = getml_dataset.GetMLDataset(dataframe_like)

        dataset_context: str = self._separator.join(context)
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
        artifact_path: str = self._separator.join(context)
        if self._log_as_artifact:
            with TemporaryDirectory() as temp_dir:
                filename: str = dataframelike.get_name(dataframe_like) + ".parquet"
                local_path: str = f"{temp_dir}/{filename}"
                dataframe_like.to_parquet(local_path)
                self._mlflowclient.log_artifact(
                    run_id=self._run_id,
                    local_path=local_path,
                    artifact_path=artifact_path,
                )
        if self._log_information:
            dataset: getml_dataset.GetMLDataset = getml_dataset.GetMLDataset(
                dataframe_like
            )
            self._mlflowclient.log_dict(
                self._run_id, dataset.as_dict(), f"{artifact_path}/{dataset.name}.json"
            )

    def _log_subset(self, subset: getml.data.Subset, context: List[str]) -> None:
        self._log_dataframe_like(subset.population, context + ["Population"])

        for name, table in subset.peripheral.items():
            self._log_dataframe_like(table, context + ["Peripheral", name])
