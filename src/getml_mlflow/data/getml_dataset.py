import json
from typing import Any, Dict, Optional, Union

import getml
from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.types import ColSpec, Schema
from typing_extensions import override


class GetMLDatasetSource(DatasetSource):
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


class GetMLDataset(Dataset):
    def __init__(
        self,
        dataframe_like: Union[getml.DataFrame, getml.data.View],
        source: Optional[DatasetSource] = None,
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ):
        self._dataframe_like = dataframe_like
        self._name: str = (
            name if name is not None else self._resolve_name(dataframe_like)
        )
        resolved_source = (
            source if source is not None else self._resolve_source(dataframe_like)
        )
        super().__init__(resolved_source, self._name, digest)

    @override
    def _compute_digest(self) -> str:
        print("GetMLDataset._compute_digest")
        return str(self._dataframe_like.name)

    @override
    def to_dict(self) -> Dict[str, str]:
        result = super().to_dict()
        if self.schema:
            schema = json.dumps({"mlflow_colspec": self.schema.to_dict()})
            result.update(
                {
                    "schema": schema,
                }
            )
        result.update(
            {
                "profile": json.dumps(self.profile),
            }
        )
        return result

    @property
    @override
    def profile(self) -> Optional[Any]:
        print("GetMLDataset.profile")
        return {"something": "else", "number_of_rows": 42, "mean": "very"}

    @property
    @override
    def schema(self) -> Optional[Any]:
        print("GetMLDataset.schema")
        return Schema([ColSpec("integer", "a"), ColSpec("float", "b")])

    def _resolve_name(
        self, dataframe_like: Union[getml.DataFrame, getml.data.View]
    ) -> str:
        if isinstance(dataframe_like, getml.DataFrame):
            return str(dataframe_like.name)
        else:
            return f"{dataframe_like.base.name}.{dataframe_like.name}"

    def _resolve_source(
        self, dataframe_like: Union[getml.DataFrame, getml.data.View]
    ) -> DatasetSource:
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


def from_getml(
    dataframe_like: Union[getml.DataFrame, getml.data.View],
    source: Optional[DatasetSource] = None,
    name: Optional[str] = None,
    digest: Optional[str] = None,
) -> GetMLDataset:
    return GetMLDataset(dataframe_like, source, name, digest)
