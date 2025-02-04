import hashlib
import json
import pathlib
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Union

import mlflow
import mlflow.data.dataset
import mlflow.data.dataset_source
import mlflow.entities.dataset
import mlflow.types
from mlflow.types.schema import ColSpec, DataType, Schema
from typing_extensions import override

import getml
from getml.data.roles.container import Roles
from getml.data.roles.types import Role
from getml_mlflow.data import dataframelike
from getml_mlflow.data.dataframelike import DataFrameLike, get_base, get_dataframe_name


class GetMLDatasetSource(mlflow.data.dataset_source.DatasetSource):
    @classmethod
    def from_parquet(
        cls,
        path: str,
        roles: Union[Dict[Union[Role, str], Iterable[str]], Roles, None] = None,
        ignore: bool = False,
        colnames: Iterable[str] = (),
    ) -> "GetMLDatasetSource":
        dataframe: getml.DataFrame = getml.DataFrame.from_parquet(
            path, pathlib.Path(path).stem, roles, ignore, False, colnames
        )
        return cls(dataframe)

    @classmethod
    def from_getml(
        cls,
        dataframe_name: str,
        roles: Union[Dict[Union[Role, str], Iterable[str]], Roles, None] = None,
    ) -> "GetMLDatasetSource":
        dataframe: getml.DataFrame = getml.DataFrame(dataframe_name, roles).load()
        return cls(dataframe)

    @classmethod
    def from_dataframe_like(cls, dataframe_like: DataFrameLike) -> "GetMLDatasetSource":
        dataframe: getml.DataFrame = get_base(dataframe_like).save().load()
        return cls(dataframe)

    def __init__(self, dataframe: getml.DataFrame) -> None:
        self._dataframe: getml.DataFrame = dataframe
        super().__init__()

    @override
    @staticmethod
    def _get_source_type() -> str:
        return "http"

    @override
    def load(self) -> Any:
        return self._dataframe.save().load()

    @override
    @staticmethod
    def _can_resolve(raw_source: Any) -> bool:
        return isinstance(raw_source, Union[str, DataFrameLike])

    @override
    @classmethod
    def _resolve(cls, raw_source: Any) -> "GetMLDatasetSource":
        if isinstance(raw_source, DataFrameLike):
            return cls.from_dataframe_like(raw_source)
        if isinstance(raw_source, str):
            if raw_source.endswith(".parquet"):
                return cls.from_parquet(raw_source)

            return cls.from_getml(raw_source)
        raise NotImplementedError(f"Cannot resolve source {raw_source}")

    @override
    def to_dict(self) -> dict:
        project_name: str = getml.project.name
        dataframe_name: str = get_dataframe_name(self._dataframe)

        return {
            "url": f"http://localhost:1709/#/getdataframe/{project_name}/{dataframe_name}/",
            "dataframe_name": dataframe_name,
            "project_name": project_name,
            "roles": self._dataframe.roles.to_dict(),
        }

    @override
    @classmethod
    def from_dict(cls, source_dict: dict) -> "GetMLDatasetSource":
        return cls.from_getml(source_dict["dataframe_name"], source_dict["roles"])


class GetMLDataset(mlflow.data.dataset.Dataset):
    GETML_ROLE_TO_MLFLOW_TYPE = {
        "categorical": DataType.string,
        "join_key": DataType.string,
        "numerical": DataType.double,
        "target": DataType.double,
        "text": DataType.string,
        "time_stamp": DataType.double,
        "unused_float": DataType.double,
        "unused_string": DataType.string,
    }

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
        return f"{dataframe_like_hash.hexdigest()[:8]}-{now_hash.hexdigest()[:8]}"

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

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "digest": self.digest,
            "source": self.source.to_dict(),
            "source_type": self.source._get_source_type(),
            "schema": self.schema.to_dict() if self.schema else None,
            "profile": self.profile,
            "roles": self._roles,
        }

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
            type=self.GETML_ROLE_TO_MLFLOW_TYPE[type],
            name=name,
            required=not type.startswith("unused_"),
        )

    def _resolve_source(
        self, dataframe_like: DataFrameLike
    ) -> mlflow.data.dataset_source.DatasetSource:
        return GetMLDatasetSource.from_dataframe_like(dataframe_like)

    @property
    def _roles(self) -> Dict[str, str]:
        return {
            key: str(value)
            for (key, value) in self._dataframe_like.roles.to_mapping().items()
        }
