from __future__ import annotations

import hashlib
import json
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import getml
from getml.data import DataFrame, Roles
from getml.data.roles.types import Role
from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.types.schema import ColSpec, DataType, Schema
from typing_extensions import override

from getml_mlflow.constants import DEFAULT_GETML_HOST, DEFAULT_GETML_PORT
from getml_mlflow.data import dataframelike
from getml_mlflow.data.dataframelike import DataFrameLike, DataFrameLikeT


class GetMLDatasetSource(DatasetSource):
    @classmethod
    def from_parquet(
        cls,
        path: str,
        roles: Union[Dict[Union[Role, str], Iterable[str]], Roles, None] = None,
        ignore: bool = False,
        colnames: Iterable[str] = (),
    ) -> GetMLDatasetSource:
        dataframe: DataFrame = DataFrame.from_parquet(
            path, Path(path).stem, roles, ignore, False, colnames
        )
        return cls(dataframe)

    @classmethod
    def from_getml(
        cls,
        dataframe_name: str,
        roles: Union[Dict[Union[Role, str], Iterable[str]], Roles, None] = None,
    ) -> GetMLDatasetSource:
        dataframe: DataFrame = DataFrame(dataframe_name, roles).load()
        return cls(dataframe)

    @classmethod
    def from_dataframe_like(cls, dataframe_like: DataFrameLike) -> GetMLDatasetSource:
        dataframe: DataFrame = dataframelike.get_base(dataframe_like).save().load()
        return cls(dataframe)

    def __init__(self, dataframe: DataFrame) -> None:
        self._dataframe: DataFrame = dataframe
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
        return isinstance(raw_source, (str, DataFrameLikeT))

    @override
    @classmethod
    def _resolve(cls, raw_source: Any) -> GetMLDatasetSource:
        if isinstance(raw_source, DataFrameLikeT):
            return cls.from_dataframe_like(raw_source)
        if isinstance(raw_source, str):
            if raw_source.endswith(".parquet"):
                return cls.from_parquet(raw_source)

            return cls.from_getml(raw_source)
        raise NotImplementedError(f"Cannot resolve source {raw_source}")

    @override
    def to_dict(self) -> dict:
        project_name: str = getml.project.name
        dataframe_name: str = dataframelike.get_dataframe_name(self._dataframe)

        return {
            # TODO: Extract URL to a default URL and make it configurable
            "url": f"http://{DEFAULT_GETML_HOST}:{DEFAULT_GETML_PORT}/#/getdataframe/{project_name}/{dataframe_name}/",
            "dataframe_name": dataframe_name,
            "project_name": project_name,
            "roles": self._dataframe.roles.to_dict(),
        }

    @override
    @classmethod
    def from_dict(cls, source_dict: dict) -> GetMLDatasetSource:
        return cls.from_getml(source_dict["dataframe_name"], source_dict["roles"])


class GetMLDataset(Dataset):
    GETML_ROLE_TO_MLFLOW_TYPE: Dict[str, DataType] = {
        "categorical": DataType.string,
        "join_key": DataType.string,
        "numerical": DataType.double,
        "target": DataType.double,
        "text": DataType.string,
        "time_stamp": DataType.double,
        "unused_float": DataType.double,
        "unused_string": DataType.string,
    }

    GETML_ROLE_TO_EMOJI: Dict[str, str] = {
        "categorical": "🗃",
        "join_key": "🔗",
        "numerical": "🔢",
        "target": "🎯",
        "text": "📝",
        "time_stamp": "⏰",
        "unused_float": "🧮",
        "unused_string": "🧵",
    }

    def __init__(
        self,
        dataframe_like: DataFrameLike,
        source: Optional[DatasetSource] = None,
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ) -> None:
        self._dataframe_like = dataframe_like
        self._name: str = (
            name if name is not None else dataframelike.get_name(dataframe_like)
        )
        resolved_source: DatasetSource = (
            source if source is not None else self._resolve_source(dataframe_like)
        )
        super().__init__(resolved_source, self._name, digest)

    @override
    def _compute_digest(self) -> str:
        return hashlib.md5(json.dumps(self._as_base_dict()).encode()).hexdigest()[:8]

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

    def _as_base_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source.to_dict(),
            "source_type": self.source._get_source_type(),
            "schema": self.schema.to_dict() if self.schema else None,
            "profile": self.profile,
            "roles": self._roles,
        }

    def as_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = self._as_base_dict()
        result.update(
            {
                "digest": self.digest,
            }
        )
        return result

    @property
    @override
    def profile(self) -> Optional[Any]:
        ncols: int = self._dataframe_like.ncols()
        nrows: Union[int, str] = self._dataframe_like.nrows()
        return {
            "num_rows": nrows,
            "num_cols": ncols,
            "num_elements": nrows if isinstance(nrows, str) else nrows * ncols,
            "base_name": dataframelike.get_dataframe_name(self._dataframe_like),
        }

    @property
    @override
    def schema(self) -> Optional[Any]:
        return Schema(
            [
                self._to_colspec(name, role)
                for (name, role) in self._dataframe_like.roles.to_mapping().items()
            ]
        )

    def _to_colspec(self, name: str, role: str) -> ColSpec:
        return ColSpec(
            type=self.GETML_ROLE_TO_MLFLOW_TYPE[role],
            name=f"{self.GETML_ROLE_TO_EMOJI[role]} {name}",
            required=not role.startswith("unused_"),
        )

    def _resolve_source(self, dataframe_like: DataFrameLike) -> DatasetSource:
        return GetMLDatasetSource.from_dataframe_like(dataframe_like)

    @cached_property
    def _roles(self) -> Dict[str, str]:
        return {
            key: str(value)
            for (key, value) in self._dataframe_like.roles.to_mapping().items()
        }
