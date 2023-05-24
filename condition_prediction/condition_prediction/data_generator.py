import numpy as np
import keras
import pandas as pd
from typing import Optional
from condition_prediction.constants import *
import tensorflow as tf

# Things this class should do
# Take in dataframes and optionally the fingerpints
# Generate a fingerprint on the lfy


class FingerprintDataGenerator(keras.utils.Sequence):
    "Generates data for Keras"

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
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        if self.fp is not None:
            return int(np.floor(self.fp.shape[0] / self.batch_size))
        return int(np.floor(len(self.data.shape[0]) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Get fingerprints
        if self.fp is not None:
            batch_product_fp = self.fp[indexes, : self.fp.shape[1] // 2]
            batch_rxn_diff_fp = self.fp[indexes, self.fp.shape[1] // 2 :]

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
        self.indexes = np.arange(len(self))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load("data/" + ID + ".npy")

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
