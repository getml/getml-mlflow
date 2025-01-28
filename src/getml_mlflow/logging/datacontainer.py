from typing import Dict, Sequence, Union

import getml
import mlflow
import mlflow.data.pandas_dataset


class DataContainerLogger:
    def __init__(self, mlflowclient: mlflow.MlflowClient, run_id: str):
        self._mlflowclient: mlflow.MlflowClient = mlflowclient
        self._run_id: str = run_id

    def log_data_containers(
        self,
        data_containers: Union[
            Sequence[Union[getml.DataFrame, getml.data.View]],
            Dict[str, Union[getml.DataFrame, getml.data.View]],
        ],
        prefix: str,
    ) -> None:
        if isinstance(data_containers, dict):
            for name, data_container in data_containers.items():
                self.log_data_container(data_container, f"{prefix}.{name}")
        else:
            for id, data_container in enumerate(data_containers):
                self.log_data_container(data_container, f"{prefix}.{id}")

    def log_data_container(
        self,
        data_container: Union[getml.DataFrame, getml.data.View, getml.data.Subset],
        context: str,
    ) -> None:
        if isinstance(data_container, (getml.DataFrame, getml.data.View)):
            self._log_dataframelike(data_container, context)
        elif isinstance(data_container, getml.data.Subset):
            self._log_subset(data_container, f"{context}")

    def _log_dataframelike(
        self, dataframelike: Union[getml.DataFrame, getml.data.View], context: str
    ) -> None:
        # name: str = str(
        #     table.name
        #     if isinstance(table, getml.DataFrame)
        #     else f"{table.name}.{table.base.name}"
        # )
        # dataset: PandasDataset = mlflow.data.pandas_dataset.from_pandas(
        #     table.to_pandas(), name=name
        # )
        self._mlflowclient.log_inputs(
            run_id=self._run_id,
            datasets=[],  # from_getml(dataframe_like)],
            # tags={MLFLOW_DATASET_CONTEXT: context},
        )

    def _log_subset(self, subset: getml.data.Subset, context: str) -> None:
        self._log_dataframelike(subset.population, f"{context}.Population")

        for name, table in subset.peripheral.items():
            self._log_dataframelike(table, f"{context}.Peripheral.{name}")
