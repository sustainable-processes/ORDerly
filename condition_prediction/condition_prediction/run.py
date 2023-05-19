import logging
from typing import List, Dict, Tuple, Optional
import dataclasses
import datetime
import pathlib
import click

LOG = logging.getLogger(__name__)

from click_loglevel import LogLevel

import tqdm
import tqdm.contrib.logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import condition_prediction.learn.ohe
import condition_prediction.learn.util

import condition_prediction.model


@dataclasses.dataclass(kw_only=True)
class ConditionPrediction:
    """
    Class for training a condition prediction model.

    1) Get the data ready for modelling
    1.1) Inputs: concat([rxn_diff_fp, product_fp])
    1.2) Targets: OHE

    """

    train_data_path: pathlib.Path
    test_data_path: pathlib.Path
    train_fp_path: pathlib.Path
    test_fp_path: pathlib.Path
    output_folder_path: pathlib.Path
    train_fraction: float
    train_val_split: float

    def __post_init__(self) -> None:
        pass

    def train_model_arguments(self) -> None:
        train_df = pd.read_parquet(self.train_data_path)
        test_df = pd.read_parquet(self.test_data_path)
        train_fp = np.load(self.train_fp_path)
        test_fp = np.load(self.test_fp_path)
        # TODO: Drop columns we don't need
        ConditionPrediction.train_model(
            train_df,
            test_df,
            train_fp,
            test_fp,
            self.output_folder_path,
            self.train_fraction,
            self.train_val_split,
        )

    @staticmethod
    def train_model(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_fp: np.ndarray,
        test_fp: np.ndarray,
        output_folder_path,
        train_fraction: float = 1.0,
        train_val_split: float = 0.8,
    ) -> None:
        """
        catalyst_in_data: bool, to determine whether we're predicting agent_002 or catalyst_000
        """
        assert train_df.shape[1] == test_df.shape[1]
        df = pd.concat([train_df, test_df], axis=0)
        df = df.reset_index(drop=True)
        test_indexes = np.arange(train_df.shape[0], df.shape[0])

        rng = np.random.default_rng(12345)

        train_indexes = np.arange(train_df.shape[0])
        rng.shuffle(train_indexes)
        train_indexes = train_indexes[: int(train_indexes.shape[0] * train_fraction)]
        train_df = train_df.iloc[train_indexes]
        train_idx = train_indexes[: int(train_indexes.shape[0] * train_val_split)]
        val_idx = train_indexes[int(train_indexes.shape[0] * train_val_split) :]

        fp_length = train_fp.shape[1]

        train_product_fp = tf.convert_to_tensor(
            train_fp[train_idx][: int(fp_length / 2)]
        )
        train_rxn_diff_fp = tf.convert_to_tensor(
            train_fp[train_idx][int(fp_length / 2) :]
        )
        val_product_fp = tf.convert_to_tensor(train_fp[val_idx][: int(fp_length / 2)])
        val_rxn_diff_fp = tf.convert_to_tensor(train_fp[val_idx][int(fp_length / 2) :])

        # If catalyst_000 exists, this means we had trust_labelling = True, and we need to recast the columns to standardise the data
        if "catalyst_000" in df.columns:
            df["agent_000"] = df["catalyst_000"]
            df["agent_001"] = df["reagent_000"]
            df["agent_002"] = df["reagent_001"]
            df.drop(
                columns=["catalyst_000", "reagent_000", "reagent_001"], inplace=True
            )

        # Get target variables ready for modelling
        (
            train_solvent_0,
            val_solvent_0,
            sol0_enc,
        ) = condition_prediction.learn.ohe.apply_train_ohe_fit(
            df[["solvent_000"]].fillna("NULL"),
            train_idx,
            val_idx,
            tensor_func=tf.convert_to_tensor,
        )
        (
            train_solvent_1,
            val_solvent_1,
            sol1_enc,
        ) = condition_prediction.learn.ohe.apply_train_ohe_fit(
            df[["solvent_001"]].fillna("NULL"),
            train_idx,
            val_idx,
            tensor_func=tf.convert_to_tensor,
        )
        (
            train_agent_0,
            val_agent_0,
            reag0_enc,
        ) = condition_prediction.learn.ohe.apply_train_ohe_fit(
            df[["agent_000"]].fillna("NULL"),
            train_idx,
            val_idx,
            tensor_func=tf.convert_to_tensor,
        )
        (
            train_agent_1,
            val_agent_1,
            reag1_enc,
        ) = condition_prediction.learn.ohe.apply_train_ohe_fit(
            df[["agent_001"]].fillna("NULL"),
            train_idx,
            val_idx,
            tensor_func=tf.convert_to_tensor,
        )
        (
            train_agent_2,
            val_agent_2,
            reag2_enc,
        ) = condition_prediction.learn.ohe.apply_train_ohe_fit(
            df[["agent_001"]].fillna("NULL"),
            train_idx,
            val_idx,
            tensor_func=tf.convert_to_tensor,
        )
        del train_df
        del test_df
        del df
        LOG.info("Data ready for modelling")

        x_train_data = (
            train_product_fp,
            train_rxn_diff_fp,
            train_solvent_0,
            train_solvent_1,
            train_agent_0,
            train_agent_1,
            train_agent_2,
        )

        x_train_eval_data = (
            train_product_fp,
            train_rxn_diff_fp,
        )

        y_train_data = (
            train_solvent_0,
            train_solvent_1,
            train_agent_0,
            train_agent_1,
            train_agent_2,
        )

        x_val_data = (
            val_product_fp,
            val_rxn_diff_fp,
            val_solvent_0,
            val_solvent_1,
            val_agent_0,
            val_agent_1,
            val_agent_2,
        )

        x_val_eval_data = (
            val_product_fp,
            val_rxn_diff_fp,
        )

        y_val_data = (
            val_solvent_0,
            val_solvent_1,
            val_agent_0,
            val_agent_1,
            val_agent_2,
        )

        train_mode = condition_prediction.model.HARD_SELECTION

        model = condition_prediction.model.build_teacher_forcing_model(
            pfp_len=train_product_fp.shape[-1],
            rxnfp_len=train_rxn_diff_fp.shape[-1],
            s1_dim=train_solvent_0.shape[-1],
            s2_dim=train_solvent_1.shape[-1],
            a1_dim=train_agent_0.shape[-1],
            a2_dim=train_agent_1.shape[-1],
            a3_dim=train_agent_2.shape[-1],
            N_h1=1024,
            N_h2=100,
            l2v=0,  # TODO check what coef they used
            mode=train_mode,
            dropout_prob=0.2,
            use_batchnorm=True,
        )

        # we use a separate model for prediction because we use a recurrent setup for prediction
        # the pred model is only different after the first component (s1)
        pred_model = condition_prediction.model.build_teacher_forcing_model(
            pfp_len=train_product_fp.shape[-1],
            rxnfp_len=train_rxn_diff_fp.shape[-1],
            s1_dim=train_solvent_0.shape[-1],
            s2_dim=train_solvent_1.shape[-1],
            a1_dim=train_agent_0.shape[-1],
            a2_dim=train_agent_1.shape[-1],
            a3_dim=train_agent_2.shape[-1],
            N_h1=1024,
            N_h2=100,
            l2v=0,
            mode=condition_prediction.model.HARD_SELECTION,
            dropout_prob=0.2,
            use_batchnorm=True,
        )

        model.compile(
            loss=[
                tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            ],
            loss_weights=[1, 1, 1, 1, 1],
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics={
                "s1": [
                    "acc",
                    tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
                    tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
                ],
                "s2": [
                    "acc",
                    tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
                    tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
                ],
                "a1": [
                    "acc",
                    tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
                    tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
                ],
                "a2": [
                    "acc",
                    tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
                    tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
                ],
                "a3": [
                    "acc",
                    tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
                    tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
                ],
            },
        )

        condition_prediction.model.update_teacher_forcing_model_weights(
            update_model=pred_model, to_copy_model=model
        )

        h = model.fit(
            x=x_train_data
            if train_mode == condition_prediction.model.TEACHER_FORCE
            else x_train_eval_data,
            y=y_train_data,
            epochs=20,
            verbose=1,
            batch_size=1024,
            validation_data=(
                x_val_data
                if train_mode == condition_prediction.model.TEACHER_FORCE
                else x_val_eval_data,
                y_val_data,
            ),
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    log_dir=condition_prediction.learn.util.log_dir(
                        prefix="TF_", comment="_MOREDATA_REG_HARDSELECT"
                    )
                ),
            ],
        )
        condition_prediction.model.update_teacher_forcing_model_weights(
            update_model=pred_model, to_copy_model=model
        )

        plt.plot(h.history["loss"], label="loss")
        plt.plot(h.history["val_loss"], label="val_loss")
        plt.legend()


@click.command()
@click.option(
    "--train_data_path",
    type=str,
    default="/data/orderly/datasets/orderly_no_trust_with_map_train.parquet",
    show_default=True,
    help="The filepath where the training data is found",
)
@click.option(
    "--test_data_path",
    default="no_trust_with_map_model",
    type=str,
    help="The filepath where the test data is found",
)
@click.option(
    "--output_folder_path",
    default="/data/orderly/datasets/orderly_no_trust_with_map_train.parquet",
    type=str,
    help="The filepath where the test data is found",
)
@click.option(
    "--train_fraction",
    default=1.0,
    type=float,
    help="The fraction of the train data that will actually be used for training (ignore the rest)",
)
@click.option(
    "--train_val_split",
    default=0.8,
    type=float,
    help="The fraction of the train data that is used for training (the rest is used for validation)",
)
@click.option(
    "--overwrite",
    type=bool,
    default=False,
    show_default=True,
    help="If true, will overwrite the contents in the output folder, else will through an error if the folder is not empty",
)
@click.option(
    "--log_file",
    type=str,
    default="default_path_model.log",
    show_default=True,
    help="path for the log file for model",
)
@click.option("--log-level", type=LogLevel(), default=logging.INFO)
def main_click(
    train_data_path: pathlib.Path,
    test_data_path: pathlib.Path,
    output_folder_path: pathlib.Path,
    train_fraction: float,
    train_val_split: float,
    overwrite: bool,
    log_file: pathlib.Path = pathlib.Path("model.log"),
    log_level: int = logging.INFO,
) -> None:
    """
    After extraction and cleaning of ORD data, this will train a condition prediction model.
    """

    _log_file = pathlib.Path(output_folder_path) / f"model.log"
    if log_file != "default_path_model.log":
        _log_file = pathlib.Path(log_file)

    main(
        train_data_path=pathlib.Path(train_data_path),
        test_data_path=pathlib.Path(test_data_path),
        output_folder_path=pathlib.Path(output_folder_path),
        train_fraction=train_fraction,
        train_val_split=train_val_split,
        overwrite=overwrite,
        log_file=_log_file,
        log_level=log_level,
    )


def main(
    train_data_path: pathlib.Path,
    test_data_path: pathlib.Path,
    output_folder_path: pathlib.Path,
    train_fraction: float,
    train_val_split: float,
    overwrite: bool,
    log_file: pathlib.Path = pathlib.Path("plots.log"),
    log_level: int = logging.INFO,
) -> None:
    """
    After extraction and cleaning of ORD data, this will train a condition prediction model.


    Functionality:
    1) Load the the train and test data
    2) Get the fingerprint to use as input (Morgan fp from rdkit). Generating FP is slow, so we do it once and save it.
    3) Apply OHE to the target variables
    3) Train the model (use tqdm to show progress)
        3.1) Save graphs of training & validation loss and accuracy
        3.2) Save the model
    4) Evaluate the model on the test data
        4.1) Save the test loss and accuracy in a log file

    We can then use the model to predict the condition of a reaction.

    """
    start_time = datetime.datetime.now()
    
    output_folder_path.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        # Assert that the output_folder_path is empty
        assert (
            len(list(output_folder_path.iterdir())) == 0
        ), f"{output_folder_path} is not empty"

    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        encoding="utf-8",
        format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=log_level,
    )

    if not isinstance(train_data_path, pathlib.Path):
        e = ValueError(f"Expect pathlib.Path: got {type(train_data_path)}")
        LOG.error(e)
        raise e
    if not isinstance(test_data_path, pathlib.Path):
        e = ValueError(f"Expect pathlib.Path: got {type(test_data_path)}")
        LOG.error(e)
        raise e
    if not isinstance(output_folder_path, pathlib.Path):
        e = ValueError(f"Expect pathlib.Path: got {type(test_data_path)}")
        LOG.error(e)
        raise e

    
    fp_directory = train_data_path.parent / "fingerprints"
    fp_directory.mkdir(parents=True, exist_ok=True)
    # Define the train_fp_path
    train_fp_path = fp_directory / (train_data_path.stem + ".npy")
    test_fp_path = fp_directory / (test_data_path.stem + ".npy")

    LOG.info(f"Beginning model training, saving to {output_folder_path}")
    instance = ConditionPrediction(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        train_fp_path=train_fp_path,
        test_fp_path=test_fp_path,
        output_folder_path=output_folder_path,
        train_fraction=train_fraction,
        train_val_split=train_val_split,
    )

    instance.train_model_arguments()

    LOG.info(f"Completed model training, saving to {output_folder_path}")

    end_time = datetime.datetime.now()
    LOG.info("Training complete, duration: {}".format(end_time - start_time))
