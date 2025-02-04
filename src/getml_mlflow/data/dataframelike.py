from typing import Union

import getml

DataFrameLike = Union[getml.DataFrame, getml.data.View]


def get_name(dataframe_like: DataFrameLike) -> str:
    if isinstance(dataframe_like, getml.DataFrame):
        return str(dataframe_like.name)
    else:
        return f"{get_dataframe_name(dataframe_like)}.{dataframe_like.name}"


def get_dataframe_name(dataframe_like: DataFrameLike) -> str:
    return str(get_base(dataframe_like).name)


def get_base(dataframe_like: DataFrameLike) -> getml.DataFrame:
    if isinstance(dataframe_like, getml.DataFrame):
        return dataframe_like
    else:
        return get_base(dataframe_like.base)
