import logging
from typing import List, Optional

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
import tqdm.contrib.logging
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.rdBase import BlockLogs
from tqdm import tqdm

from condition_prediction.constants import *

LOG = logging.getLogger(__name__)


class FingerprintDataGenerator(keras.utils.Sequence):
    """Data generator for reaction condition prediction

    Args:
        mol1: ground truth solvent 1
        mol2: ground truth solvent 2
        mol3: ground truth reagent 1
        mol4: ground truth reagent 2
        mol5: ground truth reagent 3
        data: dataframe containing ground truth solvents and reagents
        fp: fingerprints if precalculated
        mode: teacher force or hard/soft selection
        batch_size: batch size
        fp_size (int): size of fingerprint if being calcuated on the fly
        shuffle (bool): shuffle data at the end of each epoch

    Notes:
        If no fingerprints are provided, they will be calculated on the fly

    """

    def __init__(
        self,
        mol1: np.ndarray,
        mol2: np.ndarray,
        mol3: np.ndarray,
        mol4: np.ndarray,
        mol5: np.ndarray,
        data: Optional[pd.DataFrame] = None,
        fp: Optional[np.ndarray] = None,
        mode: int = TEACHER_FORCE,
        batch_size=32,
        fp_size=2048,
        shuffle=True,
    ):
        "Initialization"

        self.mol1 = mol1
        self.mol2 = mol2
        self.mol3 = mol3
        self.mol4 = mol4
        self.mol5 = mol5
        self.data = data
        self.fp = fp
        if self.data is None and self.fp is None:
            raise ValueError("Must provide either data or fp")
        self.mode = mode
        self.batch_size = batch_size
        self.fp_size = fp_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        if self.fp is not None:
            return int(np.floor(self.fp.shape[0] / self.batch_size))
        return int(np.floor(self.data.shape[0] / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Get fingerprints
        if self.fp is not None:
            batch_product_fp = self.fp[indexes, : self.fp.shape[1] // 2]
            batch_rxn_diff_fp = self.fp[indexes, self.fp.shape[1] // 2 :]
        else:
            batch_product_fp, batch_rxn_diff_fp = self.get_fp(self.data.iloc[indexes])

        # Get ground truth solvents and reagents
        batch_mol1 = tf.gather(self.mol1, indexes)
        batch_mol2 = tf.gather(self.mol2, indexes)
        batch_mol3 = tf.gather(self.mol3, indexes)
        batch_mol4 = tf.gather(self.mol4, indexes)
        batch_mol5 = tf.gather(self.mol5, indexes)

        if self.mode == TEACHER_FORCE:
            X = (
                batch_product_fp,
                batch_rxn_diff_fp,
                batch_mol1,
                batch_mol2,
                batch_mol3,
                batch_mol4,
                batch_mol5,
            )
        else:
            X = (
                batch_product_fp,
                batch_rxn_diff_fp,
            )

        y = (
            batch_mol1,
            batch_mol2,
            batch_mol3,
            batch_mol4,
            batch_mol5,
        )

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        n_examples = self.fp.shape[0] if self.fp is not None else self.data.shape[0]
        self.indexes = np.arange(n_examples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_fp(self, df: pd.DataFrame):
        product_fp = self.calc_fp(df["product_000"], radius=3, nBits=self.fp_size)
        reactant_fp_0 = self.calc_fp(df["reactant_000"], radius=3, nBits=self.fp_size)
        reactant_fp_1 = self.calc_fp(df["reactant_001"], radius=3, nBits=self.fp_size)
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
        for smiles in lst:
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
