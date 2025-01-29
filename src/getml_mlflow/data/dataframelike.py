from typing import Union

import getml

DataFrameLike = Union[getml.DataFrame, getml.data.View]


def get_name(dataframe_like: DataFrameLike) -> str:
    if isinstance(dataframe_like, getml.DataFrame):
        return str(dataframe_like.name)
    else:
        return f"{dataframe_like.base.name}.{dataframe_like.name}"
