import logging
from typing import List, Dict, Tuple, Optional
import dataclasses
import datetime
import pathlib
import click
import json
from collections import Counter

LOG = logging.getLogger(__name__)

from click_loglevel import LogLevel

import tqdm
import tqdm.contrib.logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.callbacks import EarlyStopping

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
    epochs: int
    evaluate_on_test_data: bool
    early_stopping_patience: int

    def __post_init__(self) -> None:
        pass

    def run_model_arguments(self) -> None:
        train_df = pd.read_parquet(self.train_data_path)
        test_df = pd.read_parquet(self.test_data_path)
        train_fp = np.load(self.train_fp_path)
        test_fp = np.load(self.test_fp_path)
        unnecessary_columns = [
            "date_of_experiment",
            "extracted_from_file",
            "grant_date",
            "is_mapped",
            "procedure_details",
            "rxn_str",
            "rxn_time",
            "temperature",
            "yield_000",
        ]
        train_df.drop(columns=unnecessary_columns, inplace=True)
        test_df.drop(columns=unnecessary_columns, inplace=True)
        ConditionPrediction.run_model(
            train_df,
            test_df,
            train_fp,
            test_fp,
            self.output_folder_path,
            self.train_fraction,
            self.train_val_split,
            self.epochs,
            self.early_stopping_patience,
            self.evaluate_on_test_data,
        )

    @staticmethod
    def run_model(
        train_val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_val_fp: np.ndarray,
        test_fp: np.ndarray,
        output_folder_path,
        train_fraction: float = 1.0,
        train_val_split: float = 0.8,
        epochs: int = 20,
        early_stopping_patience: int = 5,
        evaluate_on_test_data: bool = False,
    ) -> None:
        """ """

        assert train_val_df.shape[1] == test_df.shape[1]
        assert train_val_fp.shape[0] == train_val_df.shape[0]
        assert test_fp.shape[0] == test_df.shape[0]

        # concat train and test df
        df = pd.concat([train_val_df, test_df], axis=0)
        df = df.reset_index(drop=True)
        test_idx = np.arange(train_val_df.shape[0], df.shape[0])

        # Get indices for train and val
        rng = np.random.default_rng(12345)
        train_val_indexes = np.arange(train_val_df.shape[0])
        rng.shuffle(train_val_indexes)
        train_val_indexes = train_val_indexes[
            : int(train_val_indexes.shape[0] * train_fraction)
        ]
        train_val_df = train_val_df.iloc[train_val_indexes]
        train_idx = train_val_indexes[
            : int(train_val_indexes.shape[0] * train_val_split)
        ]
        val_idx = train_val_indexes[int(train_val_indexes.shape[0] * train_val_split) :]

        # Apply these to the fingerprints
        train_fp = train_val_fp[train_idx]
        val_fp = train_val_fp[val_idx]

        train_product_fp = train_fp[:, : train_fp.shape[1] // 2]
        train_rxn_diff_fp = train_fp[:, train_fp.shape[1] // 2 :]

        val_product_fp = val_fp[:, : val_fp.shape[1] // 2]
        val_rxn_diff_fp = val_fp[:, val_fp.shape[1] // 2 :]

        test_product_fp = test_fp[:, : test_fp.shape[1] // 2]
        test_rxn_diff_fp = test_fp[:, test_fp.shape[1] // 2 :]

        del train_fp, val_fp, test_fp

        # If catalyst_000 exists, this means we had trust_labelling = True, and we need to recast the columns to standardise the data
        if "catalyst_000" in df.columns:  # trust_labelling = True
            trust_labelling = True
            mol_1_col = "solvent_000"
            mol_2_col = "solvent_001"
            mol_3_col = "catalyst_000"
            mol_4_col = "reagent_000"
            mol_5_col = "reagent_001"

        else:  # trust_labelling = False
            trust_labelling = False
            mol_1_col = "solvent_000"
            mol_2_col = "solvent_001"
            mol_3_col = "agent_000"
            mol_4_col = "agent_001"
            mol_5_col = "agent_002"

        # Get target variables ready for modelling
        (
            train_mol1,
            val_mol1,
            test_mol1,
            mol1_enc,
        ) = condition_prediction.learn.ohe.apply_train_ohe_fit(
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
        ) = condition_prediction.learn.ohe.apply_train_ohe_fit(
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
        ) = condition_prediction.learn.ohe.apply_train_ohe_fit(
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
        ) = condition_prediction.learn.ohe.apply_train_ohe_fit(
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
        ) = condition_prediction.learn.ohe.apply_train_ohe_fit(
            df[[mol_5_col]].fillna("NULL"),
            train_idx,
            val_idx,
            test_idx,
            tensor_func=tf.convert_to_tensor,
        )

        if evaluate_on_test_data:
            # Determine accuracy simply by predicting the top3 most likely labels
            def benchmark_top_n_accuracy(y_train, y_test, n=3):
                y_train = y_train.tolist()
                y_test = y_test.tolist()
                # find the top 3 most likely labels in train set
                label_counts = Counter(y_train)
                top_n_train_labels = [
                    label for label, count in label_counts.most_common(n)
                ]

                correct_predictions = sum(
                    test_label in top_n_train_labels for test_label in y_test
                )

                # calculate the naive_top3_accuracy
                naive_top_n_accuracy = correct_predictions / len(y_test)

                return naive_top_n_accuracy

            mol1_top1_benchmark = benchmark_top_n_accuracy(
                train_val_df[mol_1_col],
                test_df[mol_1_col],
                1,
            )
            mol2_top1_benchmark = benchmark_top_n_accuracy(
                train_val_df[mol_2_col],
                test_df[mol_2_col],
                1,
            )
            mol3_top1_benchmark = benchmark_top_n_accuracy(
                train_val_df[mol_3_col],
                test_df[mol_3_col],
                1,
            )
            mol4_top1_benchmark = benchmark_top_n_accuracy(
                train_val_df[mol_4_col],
                test_df[mol_4_col],
                1,
            )
            mol5_top1_benchmark = benchmark_top_n_accuracy(
                train_val_df[mol_5_col],
                test_df[mol_5_col],
                1,
            )

            mol1_top3_benchmark = benchmark_top_n_accuracy(
                train_val_df[mol_1_col],
                test_df[mol_1_col],
                3,
            )
            mol2_top3_benchmark = benchmark_top_n_accuracy(
                train_val_df[mol_2_col],
                test_df[mol_2_col],
                3,
            )
            mol3_top3_benchmark = benchmark_top_n_accuracy(
                train_val_df[mol_3_col],
                test_df[mol_3_col],
                3,
            )
            mol4_top3_benchmark = benchmark_top_n_accuracy(
                train_val_df[mol_4_col],
                test_df[mol_4_col],
                3,
            )
            mol5_top3_benchmark = benchmark_top_n_accuracy(
                train_val_df[mol_5_col],
                test_df[mol_5_col],
                3,
            )

            # Save the naive_top_3 benchmark to json
            benchmark_file_path = output_folder_path / "naive_top_3.json"
            benchmark_dict = {
                f"{mol_1_col}_top1": mol1_top1_benchmark,
                f"{mol_2_col}_top1": mol2_top1_benchmark,
                f"{mol_3_col}_top1": mol3_top1_benchmark,
                f"{mol_4_col}_top1": mol4_top1_benchmark,
                f"{mol_5_col}_top1": mol5_top1_benchmark,
                f"{mol_1_col}_top3": mol1_top3_benchmark,
                f"{mol_2_col}_top3": mol2_top3_benchmark,
                f"{mol_3_col}_top3": mol3_top3_benchmark,
                f"{mol_4_col}_top3": mol4_top3_benchmark,
                f"{mol_5_col}_top3": mol5_top3_benchmark,
            }

            with open(benchmark_file_path, "w") as file:
                json.dump(benchmark_dict, file)

        del train_val_df
        del test_df
        del df
        # del mol1_enc
        # del mol2_enc
        # del mol3_enc
        # del mol4_enc
        # del mol5_enc
        LOG.info("Data ready for modelling")

        x_train_data = (
            train_product_fp,
            train_rxn_diff_fp,
            train_mol1,
            train_mol2,
            train_mol3,
            train_mol4,
            train_mol5,
        )

        x_train_eval_data = (
            train_product_fp,
            train_rxn_diff_fp,
        )

        y_train_data = (
            train_mol1,
            train_mol2,
            train_mol3,
            train_mol4,
            train_mol5,
        )

        x_val_data = (
            val_product_fp,
            val_rxn_diff_fp,
            val_mol1,
            val_mol2,
            val_mol3,
            val_mol4,
            val_mol5,
        )

        x_val_eval_data = (
            val_product_fp,
            val_rxn_diff_fp,
        )

        y_val_data = (
            val_mol1,
            val_mol2,
            val_mol3,
            val_mol4,
            val_mol5,
        )

        x_test_data = (
            test_product_fp,
            test_rxn_diff_fp,
        )

        y_test_data = (
            test_mol1,
            test_mol2,
            test_mol3,
            test_mol4,
            test_mol5,
        )

        train_mode = condition_prediction.model.HARD_SELECTION

        model = condition_prediction.model.build_teacher_forcing_model(
            pfp_len=train_product_fp.shape[-1],
            rxnfp_len=train_rxn_diff_fp.shape[-1],
            mol1_dim=train_mol1.shape[-1],
            mol2_dim=train_mol2.shape[-1],
            mol3_dim=train_mol3.shape[-1],
            mol4_dim=train_mol4.shape[-1],
            mol5_dim=train_mol5.shape[-1],
            N_h1=1024,
            N_h2=100,
            l2v=0,  # TODO check what coef they used
            mode=train_mode,
            dropout_prob=0.2,
            use_batchnorm=True,
        )

        # we use a separate model for prediction because we use a recurrent setup for prediction
        # the pred model is only different after the first component (mol1)
        pred_model = condition_prediction.model.build_teacher_forcing_model(
            pfp_len=train_product_fp.shape[-1],
            rxnfp_len=train_rxn_diff_fp.shape[-1],
            mol1_dim=train_mol1.shape[-1],
            mol2_dim=train_mol2.shape[-1],
            mol3_dim=train_mol3.shape[-1],
            mol4_dim=train_mol4.shape[-1],
            mol5_dim=train_mol5.shape[-1],
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
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            metrics={
                "mol1": [
                    "acc",
                    tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
                    tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
                ],
                "mol2": [
                    "acc",
                    tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
                    tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
                ],
                "mol3": [
                    "acc",
                    tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
                    tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
                ],
                "mol4": [
                    "acc",
                    tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
                    tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
                ],
                "mol5": [
                    "acc",
                    tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
                    tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
                ],
            },
        )

        condition_prediction.model.update_teacher_forcing_model_weights(
            update_model=pred_model, to_copy_model=model
        )
        callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=condition_prediction.learn.util.log_dir(
                    prefix="TF_", comment="_MOREDATA_REG_HARDSELECT"
                )
            )
        ]
        # Define the EarlyStopping callback
        if early_stopping_patience != 0:
            early_stop = EarlyStopping(
                monitor="val_loss", patience=early_stopping_patience
            )
            callbacks.append(early_stop)

        h = model.fit(
            x=x_train_data
            if train_mode == condition_prediction.model.TEACHER_FORCE
            else x_train_eval_data,
            y=y_train_data,
            epochs=epochs,
            verbose=1,
            batch_size=512,
            validation_data=(
                x_val_data
                if train_mode == condition_prediction.model.TEACHER_FORCE
                else x_val_eval_data,
                y_val_data,
            ),
            callbacks=callbacks,
        )
        condition_prediction.model.update_teacher_forcing_model_weights(
            update_model=pred_model, to_copy_model=model
        )
        # Save the train_val_loss plot
        plt.plot(h.history["loss"], label="loss")
        plt.plot(h.history["val_loss"], label="val_loss")
        plt.legend()
        output_file_path = output_folder_path / "train_val_loss.png"
        plt.savefig(output_file_path, bbox_inches="tight", dpi=600)

        # Save the top-3 accuracy plot
        plt.clf()
        plt.plot(
            h.history["val_mol1_top3"],
            label=f"val_{mol_1_col[:-4]}{str(int(mol_1_col[-1])+1)}_top3",
        )
        plt.plot(
            h.history["val_mol2_top3"],
            label=f"val_{mol_2_col[:-4]}{str(int(mol_2_col[-1])+1)}_top3",
        )
        plt.plot(
            h.history["val_mol3_top3"],
            label=f"val_{mol_3_col[:-4]}{str(int(mol_3_col[-1])+1)}_top3",
        )
        plt.plot(
            h.history["val_mol4_top3"],
            label=f"val_{mol_4_col[:-4]}{str(int(mol_4_col[-1])+1)}_top3",
        )
        plt.plot(
            h.history["val_mol5_top3"],
            label=f"val_{mol_5_col[:-4]}{str(int(mol_5_col[-1])+1)}_top3",
        )
        plt.legend()
        output_file_path = output_folder_path / "top3_val_accuracy.png"
        plt.savefig(output_file_path, bbox_inches="tight", dpi=600)

        # Save the train and val metrics
        train_val_file_path = output_folder_path / "train_val_metrics.json"
        train_val_metrics_dict = h.history
        train_val_metrics_dict["trust_labelling"] = trust_labelling
        with open(train_val_file_path, "w") as file:
            json.dump(train_val_metrics_dict, file)

        # TODO: Save the model
        # model_save_file_path = output_folder_path / "models"
        # model.save(model_save_file_path)

        # Save the final performance on the test set
        if evaluate_on_test_data:
            # Evaluate the model on the test set

            def get_grouped_scores(y_true, y_pred, encoders):
                components_true = []
                for enc, components in zip(encoders, y_true):
                    components_true.append(enc.inverse_transform(components))
                components_true = np.concatenate(components_true, axis=1)

                components_pred = []
                for enc, components in zip(encoders, y_pred):
                    selection_idx = np.argmax(components, axis=1)
                    component_selection = np.zeros_like(components)
                    component_selection[:, selection_idx] = 1.0
                    components_pred.append(enc.inverse_transform(component_selection))
                components_pred = np.concatenate(components_pred, axis=1)
                # Inverse transform will return None for an unknown label
                # This will introduce None, where we should only have 'NULL'
                components_true = np.where(
                    components_true == None, "NULL", components_true
                )
                components_pred = np.where(
                    components_pred == None, "NULL", components_pred
                )

                sorted_arr1 = np.sort(components_true, axis=1)
                sorted_arr2 = np.sort(components_pred, axis=1)
                return (sorted_arr1 == sorted_arr2).all(axis=1)

            test_metrics = model.evaluate(x_test_data, y_test_data)
            test_metrics_dict = dict(zip(model.metrics_names, test_metrics))
            test_metrics_dict["trust_labelling"] = trust_labelling

            ### Grouped scores
            predictions = model.predict(x_test_data)

            # Solvent scores
            solvent_scores = get_grouped_scores(
                y_test_data[:2], predictions[:2], [mol1_enc, mol2_enc]
            )
            test_metrics_dict["solvent_accuracy"] = np.mean(solvent_scores)

            # 3 agents scores
            agent_scores = get_grouped_scores(
                y_test_data[2:], predictions[2:], [mol3_enc, mol4_enc, mol5_enc]
            )
            test_metrics_dict["three_agents_accuray"] = np.mean(agent_scores)

            # Overall scores
            overall_scores = np.stack([solvent_scores, agent_scores], axis=1).all(
                axis=1
            )
            test_metrics_dict["overall_accuracy"] = np.mean(overall_scores)

            # save the test metrics
            test_metrics_file_path = output_folder_path / "test_metrics.json"
            # Save the dictionary as a JSON file
            with open(test_metrics_file_path, "w") as file:
                json.dump(test_metrics_dict, file)


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
    "--epochs",
    default=20,
    type=int,
    help="The number of epochs used for training",
)
@click.option(
    "--early_stopping_patience",
    default=5,
    type=int,
    help="Number of epochs with no improvement after which training will be stopped. If 0, then early stopping is disabled.",
)
@click.option(
    "--evaluate_on_test_data",
    default=False,
    type=bool,
    help="If True, will evaluate the model on the test data",
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
    epochs: int,
    early_stopping_patience: int,
    evaluate_on_test_data: bool,
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
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        evaluate_on_test_data=evaluate_on_test_data,
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
    epochs: int,
    early_stopping_patience: int,
    evaluate_on_test_data: bool,
    overwrite: bool,
    log_file: pathlib.Path = pathlib.Path("models.log"),
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
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        evaluate_on_test_data=evaluate_on_test_data,
    )

    instance.run_model_arguments()

    LOG.info(f"Completed model training, saving to {output_folder_path}")

    end_time = datetime.datetime.now()
    LOG.info("Training complete, duration: {}".format(end_time - start_time))
