from typing import Dict, Sequence, Union

import getml
import mlflow
import mlflow.data.pandas_dataset
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT


class TableLogging:
    @classmethod
    def log_peripheral_tables(
        cls,
        peripheral_tables: Union[
            Sequence[Union[getml.DataFrame, getml.data.View]],
            Dict[str, Union[getml.DataFrame, getml.data.View]],
        ],
    ) -> None:
        if isinstance(peripheral_tables, dict):
            for name, table in peripheral_tables.items():
                cls.log_table(table, f"Peripheral.{name}")
        else:
            for id, table in enumerate(peripheral_tables):
                cls.log_table(table, f"Peripheral.{id}")

    @classmethod
    def log_table(
        cls,
        table: Union[getml.DataFrame, getml.data.View, getml.data.Subset],
        context: str,
    ) -> None:
        if isinstance(table, (getml.DataFrame, getml.data.View)):
            cls._log_dataframe_or_view(table, context)
        elif isinstance(table, getml.data.Subset):
            cls._log_subset(table, f"{context}")

    @staticmethod
    def _log_dataframe_or_view(
        table: Union[getml.DataFrame, getml.data.View], context: str
    ) -> None:
        name: str = str(
            table.name
            if isinstance(table, getml.DataFrame)
            else f"{table.name}.{table.base.name}"
        )
        dataset: PandasDataset = mlflow.data.pandas_dataset.from_pandas(
            table.to_pandas(), name=name
        )
        mlflow.log_input(dataset=dataset, tags={MLFLOW_DATASET_CONTEXT: context})

    @classmethod
    def _log_subset(cls, subset: getml.data.Subset, context: str) -> None:
        cls._log_dataframe_or_view(subset.population, f"{context}.Population")

        for name, table in subset.peripheral.items():
            cls._log_dataframe_or_view(table, f"{context}.Peripheral.{name}")
