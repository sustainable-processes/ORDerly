from typing import Dict, Set, Optional
import io
import pathlib
import pkgutil

import pandas as pd

import orderly.extract.canonicalise
from orderly.types import *


def get_solvents(path: Optional[pathlib.Path] = None) -> pd.DataFrame:
    """reads the solvent csv data stored in the package"""
    if path is None:
        data = pkgutil.get_data("orderly.data", "solvents.csv")
        data = io.BytesIO(data)  # type: ignore
        solvents = pd.read_csv(data)
    else:
        solvents = pd.read_csv(path)

    solvents["canonical_smiles"] = solvents["smiles"].apply(
        orderly.extract.canonicalise.get_canonicalised_smiles
    )
    return solvents


def get_solvents_set(path: Optional[pathlib.Path] = None) -> Set[SOLVENT]:
    solvents = get_solvents(path=path)
    return set(solvents["canonical_smiles"])


def get_solvents_dict(
    path: Optional[pathlib.Path] = None,
) -> Dict[MOLECULE_IDENTIFIER, CANON_SMILES]:
    """
    Builds a dictionary of solvents from the solvents.csv file
    """
    # TODO Check when dict is applied we use .lower()
    # I'm not sure if we actually want to do this, it can be dangerous since acronyms and metals should be case sensitive: Pc and PC are different
    solvents = get_solvents(path=path)

    def get_df(
        name: str,
        solvents_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if solvents_df is None:
            solvents_df = get_solvents()
        else:
            solvents_df = solvents_df.copy()
        df = (
            solvents_df[[name, "canonical_smiles"]]
            .dropna()
            .rename({name: "identifer"}, axis=1)
        )
        df["identifer"] = df["identifer"].str.lower()
        return df

    output: Dict[MOLECULE_IDENTIFIER, CANON_SMILES] = (
        pd.concat(
            [
                get_df(name=i, solvents_df=solvents)
                for i in ["solvent_name_1", "solvent_name_2", "solvent_name_3"]
            ],
            axis=0,
        )
        .set_index("identifer")
        .to_dict()["canonical_smiles"]
    )
    return output
