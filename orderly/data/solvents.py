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


def get_solvents_dict(
    path: typing.Optional[pathlib.Path] = None,
) -> typing.Dict[MOLECULE_IDENTIFIER, CANON_SMILES]:
    solvents = orderly.data.get_solvents(path=path)

    # Combine the lists into a sequence of key-value pairs
    key_value_pairs = zip(
        list(solvents["stenutz_name"]) + list(solvents["cosmo_name"]),
        list(solvents["canonical_smiles"]) + list(solvents["canonical_smiles"]),
    )
    # Create a dictionary from the sequence
    return dict(key_value_pairs)
