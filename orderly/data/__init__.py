import os
import sys
import typing
import io
import pathlib
import pkgutil

import pandas as pd

import orderly.extract.canonicalise
from orderly.types import *


def get_solvents(path: typing.Optional[pathlib.Path] = None) -> pd.DataFrame:
    """reads the solvent csv data stored in the package"""
    if path is None:
        data = pkgutil.get_data("orderly.data", "solvents.csv")
        return pd.read_csv(io.BytesIO(data))

    return pd.read_csv(path)


def get_solvents_set(path: typing.Optional[pathlib.Path] = None) -> typing.Set[SOLVENT]:
    solvents = orderly.data.get_solvents(path=path)
    solvents["canonical_smiles"] = solvents["smiles"].apply(
        orderly.extract.canonicalise.get_canonicalised_smiles
    )
    return set(solvents["canonical_smiles"])


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
