import logging
import multiprocessing
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.typing import NDArray
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.rdBase import BlockLogs

# from pqdm.processes import pqdm
from tqdm import tqdm

from condition_prediction.constants import HARD_SELECTION, SOFT_SELECTION, TEACHER_FORCE
from condition_prediction.utils import apply_train_ohe_fit

LOG = logging.getLogger(__name__)

AUTOTUNE = tf.data.AUTOTUNE


@dataclass(kw_only=True)
class GenerateData:
    fp_size: int
    radius: int = 3
    df: pd.DataFrame
    product_fp: Optional[NDArray[np.int64]] = None
    rxn_diff_fp: Optional[NDArray[np.int64]] = None
    mol1: NDArray[np.float32]
    mol2: NDArray[np.float32]
    mol3: NDArray[np.float32]
    mol4: NDArray[np.float32]
    mol5: NDArray[np.float32]

    # def __post_init__(self):
    #     initializer = lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)
    #     self.pool = multiprocessing.Pool(os.cpu_count(), initializer)

    def map_idx_to_data(self, idx):
        idx = idx.numpy()
        if self.product_fp is None and self.rxn_diff_fp is None:
            result = GenerateData._map_idx_to_data_gen_fp(
                self.df,
                idx,
                self.mol1,
                self.mol2,
                self.mol3,
                self.mol4,
                self.mol5,
                self.radius,
                self.fp_size,
            )
            # result = result.get()
            return result
        else:
            return self._map_idx_to_data(
                self.df,
                idx,
                self.mol1,
                self.mol2,
                self.mol3,
                self.mol4,
                self.mol5,
                self.product_fp,
                self.rxn_diff_fp,
                self.radius,
                self.fp_size,
            )

    @staticmethod
    def _map_idx_to_data(
        df,
        idx,
        mol1,
        mol2,
        mol3,
        mol4,
        mol5,
        product_fp,
        rxn_diff_fp,
        radius=3,
        fp_size=2048,
    ):
        return (
            product_fp[idx],
            rxn_diff_fp[idx],
            mol1[idx],
            mol2[idx],
            mol3[idx],
            mol4[idx],
            mol5[idx],
        )

    @staticmethod
    def _map_idx_to_data_gen_fp(
        df,
        idx,
        mol1,
        mol2,
        mol3,
        mol4,
        mol5,
        radius=3,
        fp_size=2048,
    ):
        product_fp, rxn_diff_fp = GenerateData.get_fp(
            df.iloc[idx], radius=radius, fp_size=fp_size
        )

        return (
            product_fp,
            rxn_diff_fp,
            mol1[idx],
            mol2[idx],
            mol3[idx],
            mol4[idx],
            mol5[idx],
        )

    @staticmethod
    def get_fp(df: pd.DataFrame, radius: int, fp_size: int):
        product_fp = GenerateData.calc_fps(df["product_000"], radius, fp_size)
        reactant_fp_0 = GenerateData.calc_fps(df["reactant_001"], radius, fp_size)
        reactant_fp_1 = GenerateData.calc_fps(df["reactant_001"], radius, fp_size)
        rxn_diff_fp = product_fp - reactant_fp_0 - reactant_fp_1
        return product_fp, rxn_diff_fp

    @staticmethod
    def calc_fps(smiles_list: List[str], radius: int, fp_size: int):
        block = BlockLogs()
        # fingerprints = np.array(pqdm(smiles_list, self.calc_fp, n_jobs=8))
        fingerprints = np.array(
            [GenerateData.calc_fp(smi, radius, fp_size) for smi in tqdm(smiles_list)]
        )
        return fingerprints

    @staticmethod
    def calc_fp(smiles: str, radius: int, fp_size: int):
        # convert to mol object
        if smiles == "NULL":
            return np.zeros((fp_size,), dtype=int)
        try:
            mol = Chem.MolFromSmiles(smiles)
            # We are using hashed fingerprint, becasue an unhased FP has length: 4294967295
            fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=fp_size)
            array = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, array)
            return array
        except:
            if smiles is not None:
                LOG.warning(f"Could not generate fingerprint for {smiles=}")
            return np.zeros((fp_size,), dtype=int)


def get_dataset(
    mol1: NDArray[np.float32],
    mol2: NDArray[np.float32],
    mol3: NDArray[np.float32],
    mol4: NDArray[np.float32],
    mol5: NDArray[np.float32],
    df: Optional[pd.DataFrame] = None,
    fp: Optional[NDArray[np.int64]] = None,
    # mode: int = TEACHER_FORCE,
    fp_size: int = 2048,
    shuffle: bool = True,
    batch_size: int = 512,
    shuffle_buffer_size: int = 1000,
    cache_data: bool = False,
    cache_dir: Union[str, Path] = ".tf_cache/",
    prefetch_buffer_size: Optional[int] = None,
    interleave: bool = False,
):
    """
    Get datasets

    Args:
        df: dataframe containing ground truth solvents and reagents
        fp: fingerprints
        mode: teacher force or hard/soft selection
        fp_size: size of fingerprints
        shuffle: whether to shuffle the data
        batch_size: batch size
        shuffle_buffer_size: buffer size for shuffling. Defaults to 1000.
        cache_data: whether to cache the data
        prefetch_buffer_size: buffer size for prefetching. Defaults to 5 times the batch size

    """

    # Construct outputs
    if fp is None and df is None:
        raise ValueError("Must provide either df or fp")
    elif fp is not None and df is not None and fp.shape[0] != df.shape[0]:
        raise ValueError(
            f"Fingerprint ({fp.shape}) and dataframe ({df.shape}) not the same size"
        )

    if fp is not None:
        product_fp = fp[:, : fp.shape[1] // 2]
        rxn_diff_fp = fp[:, fp.shape[1] // 2 :]
    else:
        product_fp = None
        rxn_diff_fp = None

    fp_generator = GenerateData(
        fp_size=fp_size,
        df=df,
        product_fp=product_fp,
        rxn_diff_fp=rxn_diff_fp,
        mol1=mol1,
        mol2=mol2,
        mol3=mol3,
        mol4=mol4,
        mol5=mol5,
    )

    n_items = df.shape[0] if df is not None else fp.shape[0]  # type: ignore
    dataset = tf.data.Dataset.range(n_items)  # INdex generator

    # Need to shuffle here so it doesn't try to run the expensive stuff
    # while shuffling
    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # Batch dataset
    dataset = dataset.batch(batch_size)

    # Generate the actual data
    dataset = dataset.map(
        map_func=lambda idx: tf.py_function(
            fp_generator.map_idx_to_data,
            inp=[idx],
            Tout=[tf.int64] * 2 + [tf.float32] * 5,
        ),
        # num_parallel_calls=os.cpu_count(), deterministic=False
    )

    if cache_data:
        cache_dir = Path(cache_dir)
        if not cache_dir.exists():
            cache_dir.mkdir(exist_ok=True)
            # Read through dataset once to cache it
            print("Caching dataset")
            [1 for _ in dataset.as_numpy_iterator()]
        dataset = dataset.cache(filename=str(cache_dir / "fps"))

    if interleave:
        dataset = tf.data.Dataset.range(len(dataset)).interleave(
            lambda _: dataset,
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
            # cycle_length=16,
        )

    if prefetch_buffer_size is None:
        prefetch_buffer_size = AUTOTUNE
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
    print("Prefetch buffer size:", prefetch_buffer_size)
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
    # mode: int = TEACHER_FORCE,
    batch_size: int = 512,
    shuffle_buffer_size: int = 1000,
    prefetch_buffer_size: Optional[int] = None,
    cache_train_data: bool = False,
    cache_val_data: bool = False,
    cache_test_data: bool = False,
    interleave: bool = False,
    include_test: bool = True,
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
    train_dataset = get_dataset(
        train_mol1,
        train_mol2,
        train_mol3,
        train_mol4,
        train_mol5,
        df=df.iloc[train_idx],
        fp=train_val_fp[train_idx] if train_val_fp is not None else None,
        # mode=train_mode,
        fp_size=fp_size,
        shuffle=True,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        cache_data=cache_train_data,
        prefetch_buffer_size=prefetch_buffer_size,
        interleave=interleave,
        cache_dir=".tf_cache_train/",
    )
    val_dataset = get_dataset(
        val_mol1,
        val_mol2,
        val_mol3,
        val_mol4,
        val_mol5,
        df=df.iloc[val_idx],
        fp=train_val_fp[val_idx] if train_val_fp is not None else None,
        # mode=val_mode,
        fp_size=fp_size,
        shuffle=False,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        prefetch_buffer_size=prefetch_buffer_size,
        interleave=interleave,
        cache_data=cache_val_data,
        cache_dir=".tf_cache_val/",
    )
    if include_test:
        test_dataset = get_dataset(
            test_mol1,
            test_mol2,
            test_mol3,
            test_mol4,
            test_mol5,
            df=df.iloc[test_idx],
            fp=test_fp,
            # mode=mode,
            fp_size=fp_size,
            shuffle=False,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            prefetch_buffer_size=prefetch_buffer_size,
            interleave=interleave,
            cache_data=cache_test_data,
            cache_dir=".tf_cache_test/",
        )
    else:
        test_dataset = None

    encoders = [mol1_enc, mol2_enc, mol3_enc, mol4_enc, mol5_enc]

    return train_dataset, val_dataset, test_dataset, encoders


def _fixup_shape(X, Y):
    # ensures shape is correct after batching
    # See https://github.com/tensorflow/tensorflow/issues/32912#issuecomment-550363802
    for i in range(len(X)):
        X[i].set_shape([None, X[i].shape[1]])
    for i in range(len(Y)):
        Y[i].set_shape([None, Y[i].shape[1]])
    return X, Y


def rearrange_data_teacher_force(*data):
    X = tuple(data)
    y = tuple(data[2:])
    X, y = _fixup_shape(X, y)
    return X, y


def rearrange_data(*data):
    X = tuple(data[:2])
    y = tuple(data[2:])
    X, y = _fixup_shape(X, y)
    return X, y


def unbatch_dataset(dataset):
    X = []
    Y = []
    for i, (Xi, Yi) in dataset.enumerate().as_numpy_iterator():
        for j, xj in enumerate(Xi):
            if j > len(X) - 1:
                X.append(xj)
            else:
                X[j] = np.concatenate((X[j], xj), 0)
        for j, yj in enumerate(Yi):
            if j > len(Y) - 1:
                Y.append(yj)
            else:
                Y[j] = np.concatenate((Y[j], yj), 0)

    return X, Y
