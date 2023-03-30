import os
import sys
import typing
import io
import pathlib
import pkgutil

import pandas as pd


def get_solvents(path: typing.Optional[pathlib.Path] = None) -> pd.DataFrame:
    """reads the solvent csv data stored in the package"""
    if path is None:
        data = pkgutil.get_data("orderly.data", "solvents.csv")
        return pd.read_csv(io.BytesIO(data))

    return pd.read_csv(path)


def get_path_of_test_ords():
    return (
        pathlib.Path(os.path.dirname(sys.modules["orderly.data"].__file__))
        / "ord_test_data"
    )


def get_path_of_test_extracted_ords():
    return (
        pathlib.Path(os.path.dirname(sys.modules["orderly.data"].__file__))
        / "extracted_ord_test_data"
    )
