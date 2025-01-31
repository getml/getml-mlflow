import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import mlflow
import mlflow.data.dataset
import mlflow.data.dataset_source
import mlflow.entities.dataset
import mlflow.types
from mlflow.types.schema import ColSpec, DataType, Schema
from typing_extensions import override

from getml_mlflow.data import dataframelike
from getml_mlflow.data.dataframelike import DataFrameLike


class GetMLDatasetSource(mlflow.data.dataset_source.DatasetSource):
    @override
    @staticmethod
    def _get_source_type() -> str:
        print("GetMLDatasetSource._get_source_type")
        return "getml"

    @override
    def load(self) -> Any:
        print("GetMLDatasetSource.load")
        return None

    @override
    @staticmethod
    def _can_resolve(raw_source: Any) -> bool:
        print("GetMLDatasetSource._can_resolve")
        if isinstance(raw_source, str):
            return raw_source.startswith("getml:/")
        return False

    @override
    @classmethod
    def _resolve(cls, raw_source: Any) -> "GetMLDatasetSource":
        print("GetMLDatasetSource._resolve")
        return GetMLDatasetSource()

    @override
    def to_dict(self) -> dict:
        print("GetMLDatasetSource.to_dict")
        return {}

    @override
    @classmethod
    def from_dict(cls, source_dict: dict) -> "GetMLDatasetSource":
        print("GetMLDatasetSource.from_dict")
        return GetMLDatasetSource()


GETML_ROLE_TO_MLFLOW_TYPE = {
    "categorical": DataType.string,
    "join_key": DataType.string,
    "numerical": DataType.float,
    "target": DataType.float,
    "text": DataType.string,
    "time_stamp": DataType.float,
    "unused_float": DataType.float,
    "unused_string": DataType.string,
}


class GetMLDataset(mlflow.data.dataset.Dataset):
    def __init__(
        self,
        dataframe_like: DataFrameLike,
        source: Optional[mlflow.data.dataset_source.DatasetSource] = None,
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ):
        self._dataframe_like = dataframe_like
        self._name: str = (
            name if name is not None else dataframelike.get_name(dataframe_like)
        )
        resolved_source = (
            source if source is not None else self._resolve_source(dataframe_like)
        )
        super().__init__(resolved_source, self._name, digest)

    @override
    def _compute_digest(self) -> str:
        # TODO: Get unique identifier from dataframe_like
        datafame_like_information: List[str] = [
            dataframelike.get_name(self._dataframe_like),
            str(self._dataframe_like.ncols()),
            str(self._dataframe_like.nrows()),
        ] + self._dataframe_like.colnames
        dataframe_like_hash = hashlib.md5()
        map(dataframe_like_hash.update, map(str.encode, datafame_like_information))

        # Add current time to hash to make it unique as we don't have a unique identifier for the whole dataframe_like
        # A unique digest for a dataset is necessary as it will just be stored once in the backend
        # TODO: Add dataset_source information
        now_hash = hashlib.md5(datetime.utcnow().isoformat().encode())
        return f"{dataframe_like_hash.hexdigest()}-{now_hash.hexdigest()}"

    @override
    def to_dict(self) -> Dict[str, str]:
        result: Dict[str, str] = super().to_dict()
        result.update(
            {
                "profile": json.dumps(self.profile),
                "schema": (
                    json.dumps({"mlflow_colspec": self.schema.to_dict()})
                    if self.schema
                    else ""
                ),
            }
        )
        return result

    @property
    @override
    def profile(self) -> Optional[Any]:
        ncols: int = self._dataframe_like.ncols()
        nrows: Union[int, str] = self._dataframe_like.nrows()
        return {
            "num_rows": nrows if isinstance(nrows, str) else nrows,
            "num_cols": ncols,
            "num_elements": nrows if isinstance(nrows, str) else nrows * ncols,
        }

    @property
    @override
    def schema(self) -> Optional[Any]:
        return Schema(
            [
                self._to_colspec(name, type)
                for (name, type) in self._dataframe_like.roles.to_mapping().items()
            ]
        )

    def _to_colspec(self, name: str, type: str) -> ColSpec:
        return ColSpec(
            type=GETML_ROLE_TO_MLFLOW_TYPE[type],
            name=name,
            required=not type.startswith("unused_"),
        )

    def _resolve_source(
        self, dataframe_like: DataFrameLike
    ) -> mlflow.data.dataset_source.DatasetSource:
        # with TemporaryDirectory() as temp_dir:
        #     filename = f"{temp_dir}/{self._name}.parquet"
        #     dataframe_like.to_parquet(filename)
        #     mlflow.log_artifact(filename)  # TODO add run_id
        # artifact_uri = mlflow.get_artifact_uri(f"{self._name}.parquet")
        # for source in mlflow.data.dataset_source_registry.get_registered_sources():
        #     if type(source).__name__ == "LocalArtifactDatasetSource":
        #         source.from_dict({"uri": artifact_uri})
        # return DatasetSource.from_dict({})
        # return resolve_dataset_source(artifact_uri)
        print("GetMLDataset._resolve_source")
        return GetMLDatasetSource()


# def from_getml(
#     dataframe_like: DataFrameLike,
#     *args,
#     **kwargs,
#     # source: Optional[DatasetSource] = None,
#     # name: Optional[str] = None,
#     # digest: Optional[str] = None,
# ) -> mlflow.entities.Dataset:
#     mlflow.log_input
#     return mlflow.entities.Dataset(
#         name=dataframelike.get_name(dataframe_like),
#         digest="getml-digest",
#         source="getml-source",
#         source_type="magic-getml-source-type",
#         schema=mlflow.types.Schema([]).to_json(),
#         profile=json.dumps({"profile": "getml-profile"}),
#     )
#
