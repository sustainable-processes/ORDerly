import logging
import os
import json
from typing import List, Dict, Tuple, Optional
import dataclasses
import datetime
import pathlib
import click

from click_loglevel import LogLevel

import tqdm
import tqdm.contrib.logging
import pandas as pd
import numpy as np

from tqdm import tqdm
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.rdBase import BlockLogs
import pandas as pd
import pathlib

from orderly.types import *

LOG = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class GenerateFingerprints:
    """
    Class for generating fingerprints from a cleaned dataframe to be used for condition prediction.

    """

    clean_data_path: pathlib.Path
    fp_output_path: pathlib.Path
    fp_size: int

    def __post_init__(self) -> None:
        full_df = pd.read_parquet(self.clean_data_path)
        self.df = full_df[["product_0", "reactant_0", "reactant_1"]]
        del full_df

    def save_fingerprints(self) -> None:
        """
        Generate fingerprints from a cleaned dataframe to be used for condition prediction.
        """
        LOG.info(f"Generating fingerprints for {self.clean_data_path=}")
        product_fp, rxn_diff_fp = GenerateFingerprints.get_fp(
            self.df, fp_size=self.fp_size
        )

        fp = np.concatenate([rxn_diff_fp, product_fp], axis=1)
        LOG.info(f"Saving fingerprints to {self.fp_output_path=}")
        fp.to_parquet(self.fp_output_path)

        return

    @staticmethod
    def get_fp(
        df: pd.DataFrame,
        fp_size: int = 2048,
    ):
        product_fp = GenerateFingerprints.calc_fp(
            df["product_0"], radius=3, nBits=fp_size
        )
        reactant_fp_0 = GenerateFingerprints.calc_fp(
            df["reactant_0"], radius=3, nBits=fp_size
        )
        reactant_fp_1 = GenerateFingerprints.calc_fp(
            df["reactant_1"], radius=3, nBits=fp_size
        )

        rxn_diff_fp = product_fp - reactant_fp_0 - reactant_fp_1

        return product_fp, rxn_diff_fp

    @staticmethod
    def calc_fp(lst: List, radius: int = 3, nBits: int = 2048):
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
                ans += [array]
            except:
                LOG.warning(f"Could not generate fingerprint for {smiles=}")
                ans += [np.zeros((nBits,), dtype=int)]
        return ans


@click.command()
@click.option(
    "--clean_data_path",
    type=str,
    default="data/orderly/datasets/orderly_no_trust_with_map_train.parquet",
    show_default=True,
    help="The filepath where the cleaned data will be loaded from",
)
@click.option(
    "--fp_output_path",
    type=str,
    default="data/orderly/datasets/fp_orderly_no_trust_with_map_train.parquet",
    show_default=True,
    help="The filepath where the fp will be saved",
)
@click.option(
    "--fp_size",
    type=int,
    default=2048,
    show_default=True,
    help="Number of bits in the product fingerprint",
)
@click.option(
    "--log_file",
    type=str,
    default="default_path_fp.log",
    show_default=True,
    help="path for the log file for fp generation",
)
@click.option("--log-level", type=LogLevel(), default=logging.INFO)
def main_click(
    clean_data_path: pathlib.Path,
    fp_output_path: pathlib.Path,
    fp_size: int,
    log_file: pathlib.Path = pathlib.Path("fp.log"),
    log_level: int = logging.INFO,
) -> None:
    """
    After extraction and cleaning, this can generate the fingerprints used in the condition prediction model of the ORDerly paper.
    """
    _log_file = pathlib.Path(fp_output_path) / f"fp.log"
    if log_file != "default_path_fp.log":
        _log_file = pathlib.Path(log_file)

    main(
        clean_data_path=pathlib.Path(clean_data_path),
        fp_output_path=pathlib.Path(fp_output_path),
        fp_size=fp_size,
        log_file=_log_file,
        log_level=log_level,
    )


def main(
    clean_data_path: pathlib.Path,
    fp_output_path: pathlib.Path,
    fp_size: int,
    log_file: pathlib.Path = pathlib.Path("fp.log"),
    log_level: int = logging.INFO,
) -> None:
    """
    After extraction and cleaning, this can generate the fingerprints used in the condition prediction model of the ORDerly paper.
    """

    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        encoding="utf-8",
        format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=log_level,
    )

    if not isinstance(clean_data_path, pathlib.Path):
        e = ValueError(f"Expect pathlib.Path: got {type(clean_data_path)}")
        LOG.error(e)
        raise e
    if not isinstance(fp_output_path, pathlib.Path):
        e = ValueError(f"Expect pathlib.Path: got {type(fp_output_path)}")
        LOG.error(e)
        raise e

    fp_output_path.mkdir(parents=True, exist_ok=True)

    start_time = datetime.datetime.now()

    LOG.info(f"Beginning generation of fp for file: {clean_data_path}")
    instance = GenerateFingerprints(
        clean_data_path=clean_data_path,
        fp_output_path=fp_output_path,
        fp_size=fp_size,
    )
    instance.save_fingerprints()
    LOG.info(f"completed generation of fp, saving to {fp_output_path}")

    end_time = datetime.datetime.now()
    LOG.info("Gen fp complete, duration: {}".format(end_time - start_time))
