import dataclasses
import datetime
import json
import logging
import os
import pathlib
from typing import Dict, List, Optional, Tuple, Union

import click
import numpy as np
import pandas as pd
import tqdm
import tqdm.contrib.logging
from numpy.typing import NDArray
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.rdBase import BlockLogs
from tqdm import tqdm

from orderly.types import *

LOG = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class GenerateFingerprints:
    """
    Class for generating fingerprints from a cleaned dataframe to be used for condition prediction.

    """

    clean_data_file_path: pathlib.Path
    fp_output_path: pathlib.Path
    fp_size: int

    def __post_init__(self) -> None:
        full_df = pd.read_parquet(self.clean_data_file_path)
        self.df = full_df[["product_000", "reactant_000", "reactant_001"]]
        del full_df

    def save_fingerprints(self) -> None:
        """
        Generate fingerprints from a cleaned dataframe to be used for condition prediction.
        """
        LOG.info(f"Generating fingerprints for {self.clean_data_file_path=}")
        product_fp, rxn_diff_fp = GenerateFingerprints.get_fp(
            self.df, fp_size=self.fp_size
        )

        fp = np.concatenate([rxn_diff_fp, product_fp], axis=1)
        LOG.info(f"Saving fingerprints to {self.fp_output_path=}")

        # Save the NumPy array as .npy file
        np.save(self.fp_output_path, fp)

        return

    @staticmethod
    def get_fp(
        df: pd.DataFrame,
        fp_size: int = 2048,
    ) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        product_fp = GenerateFingerprints.calc_fp(
            df["product_000"].to_numpy(), radius=3, nBits=fp_size
        )
        reactant_fp_0 = GenerateFingerprints.calc_fp(
            df["reactant_000"].to_numpy(), radius=3, nBits=fp_size
        )
        reactant_fp_1 = GenerateFingerprints.calc_fp(
            df["reactant_001"].to_numpy(), radius=3, nBits=fp_size
        )
        rxn_diff_fp = product_fp - reactant_fp_0 - reactant_fp_1

        return product_fp, rxn_diff_fp

    @staticmethod
    def calc_fp(
        lst: NDArray[np.character], radius: int = 3, nBits: int = 2048
    ) -> NDArray[np.int64]:
        # Usage:
        # radius = 3
        # nBits = 2048
        # p0 = calc_fp(data_df['product_0'][:10000], radius=radius, nBits=nBits)
        block = BlockLogs()
        ans = []
        for smiles in tqdm(lst):
            # convert to mol object
            try:
                mol = Chem.MolFromSmiles(smiles)
                # We are using hashed fingerprint, becasue an unhased FP has length: 4294967295
                fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=nBits)
                array = np.zeros((0,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(fp, array)
                ans.append(array)
            except:
                if smiles is not None:
                    LOG.warning(f"Could not generate fingerprint for {smiles=}")
                ans.append(np.zeros((nBits,), dtype=int))
        return np.vstack(ans)


@click.command()
@click.option(
    "--clean_data_folder_path",
    type=str,
    default="data/orderly/datasets",
    show_default=True,
    help="The filepath where the cleaned data will be loaded from",
)
@click.option(
    "--fp_size",
    type=int,
    default=2048,
    show_default=True,
    help="Number of bits in the product fingerprint",
)
@click.option(
    "--overwrite",
    type=bool,
    default=False,
    show_default=True,
    help="If true, will overwrite the existing fingerprints folder",
)
def main_click(
    clean_data_folder_path: pathlib.Path,
    fp_size: int,
    overwrite: bool = False,
) -> None:
    """
    After extraction and cleaning, this can generate the fingerprints used in the condition prediction model of the ORDerly paper.
    """

    main(
        clean_data_file_path=pathlib.Path(clean_data_folder_path),
        fp_size=fp_size,
        overwrite=overwrite,
    )


def main(
    clean_data_file_path: pathlib.Path,
    fp_size: int,
    overwrite: bool = False,
    log_level: int = logging.INFO,
) -> None:
    """
    After extraction and cleaning, this can generate the fingerprints used in the condition prediction model of the ORDerly paper.
    Creates a folder inside the clean_data_folder_path called fingerprints, and loops over all the parquet files clean_data_folder_path to create an fp parquet file for each.
    """
    if not isinstance(clean_data_file_path, pathlib.Path):
        e = ValueError(f"Expect pathlib.Path: got {type(clean_data_file_path)}")
        LOG.error(e)
        raise e

    clean_data_folder_path = clean_data_file_path.parent
    # Filter the file paths to include only Parquet files

    fp_output_folder_path = pathlib.Path(clean_data_folder_path / "fingerprints")
    fp_output_folder_path.mkdir(parents=True, exist_ok=True)

    log_file = pathlib.Path(fp_output_folder_path / "fp.log")

    logging.basicConfig(
        filename=log_file,
        encoding="utf-8",
        format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=log_level,
    )

    start_time = datetime.datetime.now()
    LOG.info("Gen fp for all files in the datasets folder")

    fp_output_path = pathlib.Path(
        fp_output_folder_path / clean_data_file_path.name[:-8]
    )
    fp_output_path = fp_output_path.with_suffix(".npy")
    # assert that fp_output_path doesn't exist
    if fp_output_path.exists() and not overwrite:
        e = ValueError(
            f"{fp_output_path} already exists. Set overwrite=True to overwrite"
        )
        LOG.error(e)
        raise e

    LOG.info(f"Beginning generation of fp for file: {clean_data_file_path}")
    instance = GenerateFingerprints(
        clean_data_file_path=clean_data_file_path,
        fp_output_path=fp_output_path,
        fp_size=fp_size,
    )
    instance.save_fingerprints()
    LOG.info(f"completed generation of fp, saving to {fp_output_path}")

    end_time = datetime.datetime.now()
    LOG.info("Gen fp complete, duration: {}".format(end_time - start_time))
