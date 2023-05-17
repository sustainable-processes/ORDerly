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

from orderly.types import *

from sklearn.model_selection import train_test_split
import tensorflow as tf

import orderly.condition_prediction.reactions.get
import orderly.condition_prediction.reactions.filters
import orderly.condition_prediction.learn.ohe
import orderly.condition_prediction.learn.util

import orderly.condition_prediction.model
import orderly.condition_prediction.reactions.fingerprint


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
    model_save_path: pathlib.Path

    def __post_init__(self) -> None:
        self.train_df = pd.read_parquet(self.train_data_path)
        self.test_df = pd.read_parquet(self.test_data_path)

    def train_model(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        model_save_path,
        train_fraction: float = 1.0,
        train_val_split: float = 0.8,
    ) -> None:
        """
        catalyst_in_data: bool, to determine whether we're predicting agent_002 or catalyst_000
        """
        assert train_df.shape[1] == test_df.shape[1]

        # Make train data smaller to investigate scaling behaviour
        train_df = train_df.sample(frac=train_fraction)
        # Shuffle indices
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

        df = pd.concat([train_df, test_df])
        df.reset_index(inplace=True, drop=True)

        test_idx = df.index[len(train_df) :]
        train_idx = df.index[: int(train_df.shape[0] * train_val_split)]
        val_idx = df.index[int(train_df.shape[0] * train_val_split) :]

        # If catalyst_000 exists, this means we had trust_labelling = True, and we need to recast the columns to standardise the data
        if "catalyst_000" in df.columns:
            df["agent_000"] = df["catalyst_000"]
            df["agent_001"] = df["reagent_000"]
            df["agent_002"] = df["reagent_001"]
            df.drop(
                columns=["catalyst_000", "reagent_000", "reagent_001"], inplace=True
            )

        # Get inputs ready for modelling
        (
            product_fp,
            rxn_diff_fp,
        ) = orderly.condition_prediction.reactions.fingerprint.get_fp(
            df, model_save_path, rxn_diff_fp_size=2048, product_fp_size=2048
        )

        train_product_fp = tf.convert_to_tensor(product_fp[train_idx])
        train_rxn_diff_fp = tf.convert_to_tensor(rxn_diff_fp[train_idx])
        val_product_fp = tf.convert_to_tensor(product_fp[val_idx])
        val_rxn_diff_fp = tf.convert_to_tensor(rxn_diff_fp[val_idx])
        test_product_fp = tf.convert_to_tensor(product_fp[test_idx])
        test_rxn_diff_fp = tf.convert_to_tensor(rxn_diff_fp[test_idx])

        # Get target variables ready for modelling
        (
            train_solvent_0,
            val_solvent_0,
            sol0_enc,
        ) = orderly.condition_prediction.learn.ohe.apply_train_ohe_fit(
            df[["solvent_0"]].fillna("NULL"),
            train_idx,
            val_idx,
            tensor_func=tf.convert_to_tensor,
        )
        (
            train_solvent_1,
            val_solvent_1,
            sol1_enc,
        ) = orderly.condition_prediction.learn.ohe.apply_train_ohe_fit(
            df[["solvent_1"]].fillna("NULL"),
            train_idx,
            val_idx,
            tensor_func=tf.convert_to_tensor,
        )
        (
            train_agent_0,
            val_agent_0,
            reag0_enc,
        ) = orderly.condition_prediction.learn.ohe.apply_train_ohe_fit(
            df[["agent_000"]].fillna("NULL"),
            train_idx,
            val_idx,
            tensor_func=tf.convert_to_tensor,
        )
        (
            train_agent_1,
            val_agent_1,
            reag1_enc,
        ) = orderly.condition_prediction.learn.ohe.apply_train_ohe_fit(
            df[["agent_001"]].fillna("NULL"),
            train_idx,
            val_idx,
            tensor_func=tf.convert_to_tensor,
        )
        (
            train_agent_2,
            val_agent_2,
            reag2_enc,
        ) = orderly.condition_prediction.learn.ohe.apply_train_ohe_fit(
            df[["agent_001"]].fillna("NULL"),
            train_idx,
            val_idx,
            tensor_func=tf.convert_to_tensor,
        )

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

        train_mode = orderly.condition_prediction.model.HARD_SELECTION

        model = orderly.condition_prediction.model.build_teacher_forcing_model(
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
        pred_model = orderly.condition_prediction.model.build_teacher_forcing_model(
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
            mode=orderly.condition_prediction.model.HARD_SELECTION,
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

        orderly.condition_prediction.model.update_teacher_forcing_model_weights(
            update_model=pred_model, to_copy_model=model
        )

        h = model.fit(
            x=x_train_data
            if train_mode == orderly.condition_prediction.model.TEACHER_FORCE
            else x_train_eval_data,
            y=y_train_data,
            epochs=20,
            verbose=1,
            batch_size=1024,
            validation_data=(
                x_val_data
                if train_mode == orderly.condition_prediction.model.TEACHER_FORCE
                else x_val_eval_data,
                y_val_data,
            ),
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    log_dir=orderly.condition_prediction.learn.util.log_dir(
                        prefix="TF_", comment="_MOREDATA_REG_HARDSELECT"
                    )
                ),
            ],
        )
        orderly.condition_prediction.model.update_teacher_forcing_model_weights(
            update_model=pred_model, to_copy_model=model
        )

        plt.plot(h.history["loss"], label="loss")
        plt.plot(h.history["val_loss"], label="val_loss")
        plt.legend()


@click.command()
@click.option(
    "--log_file",
    type=str,
    default="default_path_plot.log",
    show_default=True,
    help="path for the log file for model",
)
@click.option("--log-level", type=LogLevel(), default=logging.INFO)
def main_click(
    train_data_path: pathlib.Path,
    test_data_path: pathlib.Path,
    log_file: pathlib.Path = pathlib.Path("model.log"),
    log_level: int = logging.INFO,
) -> None:
    """
    After extraction and cleaning of ORD data, this will train a condition prediction model.


    """
    _log_file = pathlib.Path(plot_output_path) / f"plot.log"
    if log_file != "default_path_plot.log":
        _log_file = pathlib.Path(log_file)

    main(
        plot_waterfall_bool=plot_waterfall_bool,
        log_file=_log_file,
        log_level=log_level,
    )


def main(
    train_data_path: pathlib.Path,
    test_data_path: pathlib.Path,
    model_save_path: pathlib.Path,
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

    model_save_path.mkdir(parents=True, exist_ok=True)

    start_time = datetime.datetime.now()

    LOG.info(f"Beginning model training, saving to {model_save_path}")
    instance = ConditionPrediction(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        model_save_path=model_save_path,
    )

    LOG.info(f"Completed model training, saving to {model_save_path}")

    end_time = datetime.datetime.now()
    LOG.info("Training complete, duration: {}".format(end_time - start_time))
