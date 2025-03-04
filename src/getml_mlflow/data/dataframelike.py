from __future__ import annotations

from typing import Union

from getml.data import DataFrame, View

DataFrameLike = Union[DataFrame, View]
DataFrameLikeT = (DataFrame, View)


def get_name(dataframe_like: DataFrameLike) -> str:
    if isinstance(dataframe_like, DataFrame):
        return str(dataframe_like.name)
    else:
        return f"{get_dataframe_name(dataframe_like)}.{dataframe_like.name}"


def get_dataframe_name(dataframe_like: DataFrameLike) -> str:
    return str(get_base(dataframe_like).name)


def get_base(dataframe_like: DataFrameLike) -> DataFrame:
    if isinstance(dataframe_like, DataFrame):
        return dataframe_like
    return get_base(dataframe_like.base)
