import logging
from typing import List, Optional

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.rdBase import BlockLogs

from condition_prediction.constants import HARD_SELECTION, SOFT_SELECTION, TEACHER_FORCE
from condition_prediction.utils import apply_train_ohe_fit

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
        n = self.data.shape[0] if self.data is not None else self.fp.shape[0]
        k = n // self.batch_size
        k += 1 if n % self.batch_size != 0 else 0
        return k

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


def get_data_generators(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    molecule_columns: List[str],
    fp_size: int = 2048,
    train_val_fp: Optional[np.ndarray] = None,
    test_fp: Optional[np.ndarray] = None,
    train_mode: int = TEACHER_FORCE,
    batch_size: int = 512,
):
    """
    Get data generators for train, val and test

    Args:
        train_val_df: dataframe containing ground truth solvents and reagents for train and val
        test_df: dataframe containing ground truth solvents and reagents for test
        train_val_fp: fingerprints for train and val
        test_fp: fingerprints for test
        train_fraction: fraction of train_val_df to use
        train_val_split: fraction of train_val_df to use for train
        train_mode: teacher force or hard/soft selection

    """
    # Get column names
    mol_1_col = molecule_columns[0]
    mol_2_col = molecule_columns[1]
    mol_3_col = molecule_columns[2]
    mol_4_col = molecule_columns[3]
    mol_5_col = molecule_columns[4]

    # Get target variables ready for modelling
    (
        train_mol1,
        val_mol1,
        test_mol1,
        mol1_enc,
    ) = apply_train_ohe_fit(
        df[[mol_1_col]].fillna("NULL"),
        train_idx,
        val_idx,
        test_idx,
        tensor_func=tf.convert_to_tensor,
    )
    (
        train_mol2,
        val_mol2,
        test_mol2,
        mol2_enc,
    ) = apply_train_ohe_fit(
        df[[mol_2_col]].fillna("NULL"),
        train_idx,
        val_idx,
        test_idx,
        tensor_func=tf.convert_to_tensor,
    )
    (
        train_mol3,
        val_mol3,
        test_mol3,
        mol3_enc,
    ) = apply_train_ohe_fit(
        df[[mol_3_col]].fillna("NULL"),
        train_idx,
        val_idx,
        test_idx,
        tensor_func=tf.convert_to_tensor,
    )
    (
        train_mol4,
        val_mol4,
        test_mol4,
        mol4_enc,
    ) = apply_train_ohe_fit(
        df[[mol_4_col]].fillna("NULL"),
        train_idx,
        val_idx,
        test_idx,
        tensor_func=tf.convert_to_tensor,
    )
    (
        train_mol5,
        val_mol5,
        test_mol5,
        mol5_enc,
    ) = apply_train_ohe_fit(
        df[[mol_5_col]].fillna("NULL"),
        train_idx,
        val_idx,
        test_idx,
        tensor_func=tf.convert_to_tensor,
    )

    # Create fingerprint generators
    train_generator = FingerprintDataGenerator(
        mol1=train_mol1,
        mol2=train_mol2,
        mol3=train_mol3,
        mol4=train_mol4,
        mol5=train_mol5,
        fp=train_val_fp[train_idx] if train_val_fp is not None else None,
        data=df.iloc[train_idx],
        mode=train_mode,
        batch_size=batch_size,
        shuffle=True,
        fp_size=fp_size,
    )
    val_mode = (
        HARD_SELECTION
        if train_mode == TEACHER_FORCE or train_mode == HARD_SELECTION
        else SOFT_SELECTION
    )
    val_generator = FingerprintDataGenerator(
        mol1=val_mol1,
        mol2=val_mol2,
        mol3=val_mol3,
        mol4=val_mol4,
        mol5=val_mol5,
        fp=train_val_fp[val_idx] if train_val_fp is not None else None,
        data=df.iloc[val_idx],
        mode=val_mode,
        batch_size=batch_size,
        shuffle=False,
        fp_size=fp_size,
    )
    test_generator = FingerprintDataGenerator(
        mol1=test_mol1,
        mol2=test_mol2,
        mol3=test_mol3,
        mol4=test_mol4,
        mol5=test_mol5,
        fp=test_fp,
        data=df.iloc[test_idx],
        mode=val_mode,
        batch_size=batch_size,
        shuffle=False,
        fp_size=fp_size,
    )

    encoders = [mol1_enc, mol2_enc, mol3_enc, mol4_enc, mol5_enc]

    return train_generator, val_generator, test_generator, encoders
