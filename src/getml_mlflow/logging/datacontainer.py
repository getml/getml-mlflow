from __future__ import annotations

from enum import Enum
from functools import cached_property
from tempfile import TemporaryDirectory
from typing import Dict, List, Sequence, Union

from getml.data import Subset
from mlflow import MlflowClient
from mlflow.entities import DatasetInput, InputTag, Run
from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT

from getml_mlflow.data import dataframelike
from getml_mlflow.data.dataframelike import DataFrameLike, DataFrameLikeT
from getml_mlflow.data.getml_dataset import GetMLDataset
from getml_mlflow.loggingconfiguration import DataContainerLoggingConfiguration


class DataContainerLoggerTarget(str, Enum):
    ARTIFACT = "artifact"
    INPUT = "input"


class DataContainerLogger:
    @classmethod
    def as_artifact(
        cls,
        mlflow_client: MlflowClient,
        run_id: str,
        *,
        logging_configuration: DataContainerLoggingConfiguration = DataContainerLoggingConfiguration(),
    ) -> DataContainerLogger:
        return cls(
            mlflow_client,
            run_id,
            DataContainerLoggerTarget.ARTIFACT,
            logging_configuration=logging_configuration,
        )

    @classmethod
    def as_input(
        cls,
        mlflow_client: MlflowClient,
        run_id: str,
        *,
        logging_configuration: DataContainerLoggingConfiguration = DataContainerLoggingConfiguration(),
    ) -> DataContainerLogger:
        return cls(
            mlflow_client,
            run_id,
            DataContainerLoggerTarget.INPUT,
            logging_configuration=logging_configuration,
        )

    def __init__(
        self,
        mlflow_client: MlflowClient,
        run_id: str,
        target: DataContainerLoggerTarget,
        *,
        logging_configuration: DataContainerLoggingConfiguration = DataContainerLoggingConfiguration(),
    ) -> None:
        self._mlflow_client: MlflowClient = mlflow_client
        self._run_id: str = run_id
        self._run: Run = self._mlflow_client.get_run(run_id)
        self._target: DataContainerLoggerTarget = target
        self._logging_configuration: DataContainerLoggingConfiguration = (
            logging_configuration
        )

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
        data_container: Union[DataFrameLike, Subset],
        context: Union[str, List[str]],
    ) -> None:
        if isinstance(context, str):
            context = [context]
        if isinstance(data_container, DataFrameLikeT):
            self._log_dataframe_like(data_container, context)
        elif isinstance(data_container, Subset):
            # Subset is a convenience case, usually used to transfer
            # population and peripheral tables together as Population
            # and then internally split into Population and Peripheral
            # Therefore we remove the last context element, usually "Population"
            self._log_subset(data_container, context[:-1])

    def _log_dataframe_like_as_input(
        self, dataframe_like: DataFrameLike, context: List[str]
    ) -> None:
        if not self._logging_configuration.log_information:
            return

        dataset: GetMLDataset = GetMLDataset(dataframe_like)

        dataset_context: str = self._separator.join(context)
        dataset_input: DatasetInput = DatasetInput(
            dataset=dataset._to_mlflow_entity(),
            tags=[InputTag(key=MLFLOW_DATASET_CONTEXT, value=dataset_context)],
        )
        self._mlflow_client.log_inputs(
            run_id=self._run_id,
            datasets=[dataset_input],
        )
        self._mlflow_client.log_dict(
            self._run_id,
            dataset.as_dict(),
            f"input/dataset/{dataset_context}.{dataset.name}.json",
        )

    def _log_dataframe_like_as_artifact(
        self, dataframe_like: DataFrameLike, context: List[str]
    ) -> None:
        artifact_path: str = self._separator.join(context)
        if self._logging_configuration.log_as_artifact:
            with TemporaryDirectory() as temp_dir:
                filename: str = dataframelike.get_name(dataframe_like) + ".parquet"
                local_path: str = f"{temp_dir}/{filename}"
                dataframe_like.to_parquet(local_path)
                self._mlflow_client.log_artifact(
                    run_id=self._run_id,
                    local_path=local_path,
                    artifact_path=artifact_path,
                )
        if self._logging_configuration.log_information:
            dataset: GetMLDataset = GetMLDataset(dataframe_like)
            self._mlflow_client.log_dict(
                self._run_id, dataset.as_dict(), f"{artifact_path}/{dataset.name}.json"
            )

    def _log_subset(self, subset: Subset, context: List[str]) -> None:
        self._log_dataframe_like(subset.population, context + ["Population"])

        for name, table in subset.peripheral.items():
            self._log_dataframe_like(table, context + ["Peripheral", name])
