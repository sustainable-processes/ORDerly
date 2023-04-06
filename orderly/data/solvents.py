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
        solvents = pd.read_csv(io.BytesIO(data))
    else:
        solvents = pd.read_csv(path)

    solvents["canonical_smiles"] = solvents["smiles"].apply(
        orderly.extract.canonicalise.get_canonicalised_smiles
    )
    return solvents


def get_solvents_set(path: typing.Optional[pathlib.Path] = None) -> typing.Set[SOLVENT]:
    solvents = orderly.data.get_solvents(path=path)
    return set(solvents["canonical_smiles"])


def get_solvents_dict(
    path: typing.Optional[pathlib.Path] = None,
) -> typing.Dict[MOLECULE_IDENTIFIER, CANON_SMILES]:
    """
    Builds a dictionary of solvents from the solvents.csv file
    """
    # TODO Check when dict is applied we use .lower()
    solvents = orderly.data.get_solvents(path=path)

    def get_df(
        name: str,
        solvents_df: typing.Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if solvents_df is None:
            solvents_df = orderly.data.get_solvents()
        else:
            solvents_df = solvents_df.copy()
        df = (
            solvents_df[[name, "canonical_smiles"]]
            .dropna()
            .rename({name: "identifer"}, axis=1)
        )
        df["identifer"] = df["identifer"].str.lower()
        return df

    return (
        pd.concat(
            [
                get_df(name=i, solvents_df=solvents)
                for i in ["stenutz_name", "cosmo_name", "other_name_1", "other_name_2"]
            ],
            axis=0,
        )
        .set_index("identifer")
        .to_dict()["canonical_smiles"]
    )
