from typing import List
import pathlib
import logging

import pandas as pd

LOG = logging.getLogger(__name__)


def save_list(x: List[str], path: pathlib.Path) -> None:
    assert isinstance(x, list)
    for i in x:
        if not isinstance(i, str):
            e = TypeError(f"expected a string but got {type(i)=} for {i=}")
            LOG.error(e)
            raise e
    pd.Series(x, dtype=str).to_csv(path, index=False)


def load_list(path: pathlib.Path) -> List[str]:
    return pd.read_csv(path).squeeze("columns").tolist()  # type: ignore
