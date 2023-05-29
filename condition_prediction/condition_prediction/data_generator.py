import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.typing import NDArray
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.rdBase import BlockLogs

from condition_prediction.constants import HARD_SELECTION, SOFT_SELECTION, TEACHER_FORCE
from condition_prediction.utils import apply_train_ohe_fit

LOG = logging.getLogger(__name__)

AUTOTUNE = tf.data.AUTOTUNE


@dataclass(kw_only=True)
class GenerateData:
    fp_size: int
    mode: int
    df: pd.DataFrame
    mol1: NDArray[np.float32]
    mol2: NDArray[np.float32]
    mol3: NDArray[np.float32]
    mol4: NDArray[np.float32]
    mol5: NDArray[np.float32]

    def map_idx_to_data(self, idx):
        idx = idx.numpy()
        product_fp, rxn_diff_fp = self.get_fp(self.df.iloc[idx], fp_size=self.fp_size)
        # product_fp = product_fp[:, np.newaxis]
        # rxn_diff_fp = rxn_diff_fp[:, np.newaxis]
        return (
            product_fp,
            rxn_diff_fp,
            self.mol1[idx],
            self.mol2[idx],
            self.mol3[idx],
            self.mol4[idx],
            self.mol5[idx],
        )

    @staticmethod
    def get_fp(row, fp_size: int = 2048):
        product_fp = GenerateData.calc_fp(row["product_000"], radius=3, nBits=fp_size)
        reactant_fp_0 = GenerateData.calc_fp(
            row["reactant_001"], radius=3, nBits=fp_size
        )
        reactant_fp_1 = GenerateData.calc_fp(
            row["reactant_001"], radius=3, nBits=fp_size
        )
        rxn_diff_fp = product_fp - reactant_fp_0 - reactant_fp_1
        return product_fp, rxn_diff_fp

    @staticmethod
    def calc_fp(smiles: str, radius: int = 3, nBits: int = 2048):
        # Usage:
        # radius = 3
        # nBits = 2048
        # p0 = calc_fp(data_df['product_0'][:10000], radius=radius, nBits=nBits)
        block = BlockLogs()
        # convert to mol object
        try:
            mol = Chem.MolFromSmiles(smiles)
            # We are using hashed fingerprint, becasue an unhased FP has length: 4294967295
            fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=nBits)
            array = np.zeros((0,), dtype=np.int64)
            DataStructs.ConvertToNumpyArray(fp, array)
            return array
        except:
            if smiles is not None:
                LOG.warning(f"Could not generate fingerprint for {smiles=}")
            return np.zeros((nBits,), dtype=np.int64)


def get_dataset(
    mol1: NDArray[np.int64],
    mol2: NDArray[np.int64],
    mol3: NDArray[np.int64],
    mol4: NDArray[np.int64],
    mol5: NDArray[np.int64],
    df: Optional[pd.DataFrame] = None,
    fp: Optional[NDArray[np.int64]] = None,
    mode: int = TEACHER_FORCE,
    fp_size: int = 2048,
    shuffle: bool = True,
    # num_parallel_calls: Optional[int] = None,
):
    # Construct outputs
    if fp is None and df is None:
        raise ValueError("Must provide either df or fp")

    if fp is None:
        fp_generator = GenerateData(
            fp_size=fp_size,
            mode=mode,
            df=df,
            mol1=mol1,
            mol2=mol2,
            mol3=mol3,
            mol4=mol4,
            mol5=mol5,
        )
        z = list(range(df.shape[0]))  # The index generator
        dataset = tf.data.Dataset.from_generator(lambda: z, tf.uint8)

        def map_func(idx):
            data = tf.py_function(
                fp_generator.map_idx_to_data,
                inp=[idx],
                Tout=[tf.int64] * 2 + [tf.float32] * 5,
            )

            if mode == TEACHER_FORCE:
                X = tuple(data)
            else:
                X = tuple(data[:2])
            y = tuple(data[2:])
            return X, y

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.map(map_func=map_func, num_parallel_calls=AUTOTUNE)
    else:
        product_fp = fp[:, : fp.shape[1] // 2]
        rxn_diff_fp = fp[:, fp.shape[1] // 2 :]
        if mode == TEACHER_FORCE:
            X = (
                product_fp,
                rxn_diff_fp,
                mol1,
                mol2,
                mol3,
                mol4,
                mol5,
            )
        else:
            X = (
                product_fp,
                rxn_diff_fp,
            )
        dataset = tf.data.Dataset.from_tensor_slices((X, y))

    return dataset


def get_datasets(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    molecule_columns: List[str],
    fp_size: int = 2048,
    train_val_fp: Optional[np.ndarray] = None,
    test_fp: Optional[np.ndarray] = None,
    train_mode: int = TEACHER_FORCE,
    # batch_size: int = 512,
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
        # tensor_func=tf.convert_to_tensor,
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
        # tensor_func=tf.convert_to_tensor,
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
        # tensor_func=tf.convert_to_tensor,
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
        # tensor_func=tf.convert_to_tensor,
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
        # tensor_func=tf.convert_to_tensor,
    )

    # Get datsets
    val_mode = (
        HARD_SELECTION
        if train_mode == TEACHER_FORCE or train_mode == HARD_SELECTION
        else SOFT_SELECTION
    )
    train_dataset = get_dataset(
        train_mol1,
        train_mol2,
        train_mol3,
        train_mol4,
        train_mol5,
        df=df.iloc[train_idx],
        fp=train_val_fp,
        mode=train_mode,
        fp_size=fp_size,
        shuffle=True,
    )
    val_dataset = get_dataset(
        val_mol1,
        val_mol2,
        val_mol3,
        val_mol4,
        val_mol5,
        df=df.iloc[val_idx],
        fp=train_val_fp,
        mode=val_mode,
        fp_size=fp_size,
        shuffle=False,
    )
    test_dataset = get_dataset(
        test_mol1,
        test_mol2,
        test_mol3,
        test_mol4,
        test_mol5,
        df=df.iloc[test_idx],
        fp=test_fp,
        mode=val_mode,
        fp_size=fp_size,
        shuffle=False,
    )

    encoders = [mol1_enc, mol2_enc, mol3_enc, mol4_enc, mol5_enc]

    return train_dataset, val_dataset, test_dataset, encoders


def configure_for_performance(ds, batch_size):
    ds = ds.cache()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds
