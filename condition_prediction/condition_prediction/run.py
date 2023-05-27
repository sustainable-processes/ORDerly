import dataclasses
import datetime
import json
import logging
import os
import pathlib
from collections import Counter
from typing import Dict, List, Optional, Tuple

import click

LOG = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import tensorflow as tf
from click_loglevel import LogLevel
from keras.callbacks import EarlyStopping
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

import wandb
from condition_prediction.constants import HARD_SELECTION, SOFT_SELECTION, TEACHER_FORCE
from condition_prediction.data_generator import get_data_generators
from condition_prediction.model import (
    build_teacher_forcing_model,
    update_teacher_forcing_model_weights,
)
from condition_prediction.utils import (
    TrainingMetrics,
    frequency_informed_accuracy,
    get_grouped_scores,
    get_random_splits,
    post_training_plots,
)


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
    generate_fingerprints: bool
    fp_size: int
    dropout: float
    hidden_size_1: int
    hidden_size_2: int
    lr: float
    batch_size: int
    workers: int
    evaluate_on_test_data: bool
    early_stopping_patience: int
    wandb_logging: bool
    wandb_project: str
    wandb_entity: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    wandb_group: Optional[str] = None

    def __post_init__(self) -> None:
        pass

    def run_model_arguments(self) -> None:
        train_df = pd.read_parquet(self.train_data_path)
        test_df = pd.read_parquet(self.test_data_path)
        train_fp = None
        test_fp = None
        if not self.generate_fingerprints:
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
        self.run_model(
            train_val_df=train_df,
            test_df=test_df,
            train_val_fp=train_fp,
            test_fp=test_fp,
            output_folder_path=self.output_folder_path,
            train_fraction=self.train_fraction,
            train_val_split=self.train_val_split,
            epochs=self.epochs,
            fp_size=self.fp_size,
            dropout=self.dropout,
            hidden_size_1=self.hidden_size_1,
            hidden_size_2=self.hidden_size_2,
            lr=self.lr,
            batch_size=self.batch_size,
            workers=self.workers,
            early_stopping_patience=self.early_stopping_patience,
            evaluate_on_test_data=self.evaluate_on_test_data,
            wandb_project=self.wandb_project,
            wandb_entity=self.wandb_entity,
            wandb_logging=self.wandb_logging,
            wandb_tags=self.wandb_tags,
            wandb_group=self.wandb_group,
        )

    @staticmethod
    def get_frequency_informed_guess(
        train_val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_folder_path: pathlib.Path,
        molecule_columns: List[str],
    ) -> None:
        mol_1_col = molecule_columns[0]
        mol_2_col = molecule_columns[1]
        mol_3_col = molecule_columns[2]
        mol_4_col = molecule_columns[3]
        mol_5_col = molecule_columns[4]

        # Evaulate whether the correct set of labels have been predicted, rather than treating them separately
        solvent_accuracy, most_common_solvents = frequency_informed_accuracy(
            (train_val_df[mol_1_col], train_val_df[mol_2_col]),
            (test_df[mol_1_col], test_df[mol_2_col]),
        )
        agent_accuracy, most_common_agents = frequency_informed_accuracy(
            (
                train_val_df[mol_3_col],
                train_val_df[mol_4_col],
                train_val_df[mol_5_col],
            ),
            (test_df[mol_3_col], test_df[mol_4_col], test_df[mol_5_col]),
        )
        overall_accuracy, most_common_combination = frequency_informed_accuracy(
            (
                train_val_df[mol_1_col],
                train_val_df[mol_2_col],
                train_val_df[mol_3_col],
                train_val_df[mol_4_col],
                train_val_df[mol_5_col],
            ),
            (
                test_df[mol_1_col],
                test_df[mol_2_col],
                test_df[mol_3_col],
                test_df[mol_4_col],
                test_df[mol_5_col],
            ),
        )

        # Save the naive_top_3 benchmark to json
        benchmark_file_path = output_folder_path / "freq_informed_acc.json"
        benchmark_dict = {
            f"most_common_solvents": most_common_solvents,
            f"most_common_agents": most_common_agents,
            f"most_common_combination": most_common_combination,
            f"solvent_acc": solvent_accuracy,
            f"agent_acc": agent_accuracy,
            f"overall_acc": overall_accuracy,
        }

        with open(benchmark_file_path, "w") as file:
            json.dump(benchmark_dict, file)

    @staticmethod
    def run_model(
        train_val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_folder_path,
        train_val_fp: Optional[np.ndarray] = None,
        test_fp: Optional[np.ndarray] = None,
        train_fraction: float = 1.0,
        train_val_split: float = 0.8,
        epochs: int = 20,
        early_stopping_patience: int = 5,
        evaluate_on_test_data: bool = False,
        train_mode: int = HARD_SELECTION,
        batch_size: int = 512,
        fp_size: int = 2048,
        dropout: float = 0.2,
        hidden_size_1: int = 1024,
        hidden_size_2: int = 100,
        lr: float = 0.01,
        workers: int = 1,
        wandb_logging: bool = True,
        wandb_project: str = "orderly",
        wandb_entity: Optional[str] = None,
        wandb_tags: Optional[List[str]] = None,
        wandb_group: Optional[str] = None,
    ) -> None:
        """
        Run condition prediction training

        """
        ### Data setup ###
        assert train_val_df.shape[1] == test_df.shape[1]

        # Concat train and test df
        df = pd.concat([train_val_df, test_df], axis=0)
        df = df.reset_index(drop=True)
        test_idx = np.arange(train_val_df.shape[0], df.shape[0])

        # Get indices for train and val
        train_idx, val_idx = get_random_splits(
            n_indices=train_val_df.shape[0],
            train_fraction=train_fraction,
            train_val_split=train_val_split,
        )

        # Apply these to the fingerprints
        if train_val_fp is not None:
            assert train_val_fp.shape[0] == train_val_df.shape[0]
            fp_size = train_val_fp.shape[1] // 2
        if test_fp is not None:
            assert test_fp.shape[0] == test_df.shape[0]

        # If catalyst_000 exists, this means we had trust_labelling = True,
        # and we need to recast the columns to standardise the data
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
        molecule_columns = [mol_1_col, mol_2_col, mol_3_col, mol_4_col, mol_5_col]

        (
            train_generator,
            val_generator,
            test_generator,
            encoders,
        ) = get_data_generators(
            df=df,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            fp_size=fp_size,
            train_val_fp=train_val_fp,
            test_fp=test_fp,
            train_mode=train_mode,
            molecule_columns=molecule_columns,
            batch_size=batch_size,
        )
        y_test_data = (
            test_generator.mol1,
            test_generator.mol2,
            test_generator.mol3,
            test_generator.mol4,
            test_generator.mol5,
        )
        if evaluate_on_test_data:
            ConditionPrediction.get_frequency_informed_guess(
                train_val_df=train_val_df,
                test_df=test_df,
                output_folder_path=output_folder_path,
                molecule_columns=molecule_columns,
            )

        del train_val_df
        del test_df
        LOG.info("Data ready for modelling")

        ### Model Setup ###
        model = build_teacher_forcing_model(
            pfp_len=fp_size,
            rxnfp_len=fp_size,
            mol1_dim=train_generator.mol1.shape[-1],
            mol2_dim=train_generator.mol2.shape[-1],
            mol3_dim=train_generator.mol3.shape[-1],
            mol4_dim=train_generator.mol4.shape[-1],
            mol5_dim=train_generator.mol5.shape[-1],
            N_h1=hidden_size_1,
            N_h2=hidden_size_2,
            l2v=0,  # TODO check what coef they used
            mode=train_mode,
            dropout_prob=dropout,
            use_batchnorm=True,
        )
        # we use a separate model for prediction because we use a recurrent setup for prediction
        # the pred model is only different after the first component (mol1)
        pred_model = build_teacher_forcing_model(
            pfp_len=fp_size,
            rxnfp_len=fp_size,
            mol1_dim=train_generator.mol1.shape[-1],
            mol2_dim=train_generator.mol2.shape[-1],
            mol3_dim=train_generator.mol3.shape[-1],
            mol4_dim=train_generator.mol4.shape[-1],
            mol5_dim=train_generator.mol5.shape[-1],
            N_h1=hidden_size_1,
            N_h2=hidden_size_2,
            l2v=0,
            mode=HARD_SELECTION,
            dropout_prob=dropout,
            use_batchnorm=True,
        )

        model.compile(
            loss=[
                tf.keras.losses.CategoricalCrossentropy(from_logits=False)
                for _ in range(5)
            ],
            loss_weights=[1, 1, 1, 1, 1],
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            metrics={
                f"mol{i}": [
                    "acc",
                    tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
                    tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
                ]
                for i in range(1, 6)
            },
        )
        update_teacher_forcing_model_weights(
            update_model=pred_model, to_copy_model=model
        )

        ### Training ###
        # callbacks = [
        #     tf.keras.callbacks.TensorBoard(
        #         log_dir=log_dir(prefix="TF_", comment="_MOREDATA_REG_HARDSELECT")
        #     )
        # ]
        callbacks = [
            TrainingMetrics(
                n_train=train_idx.shape[0],
                batch_size=train_generator.batch_size,
            )
        ]
        # Define the EarlyStopping callback
        if early_stopping_patience != 0:
            early_stop = EarlyStopping(
                monitor="val_loss", patience=early_stopping_patience
            )
            callbacks.append(early_stop)
        if wandb_logging:
            wandb_tags = [] if wandb_tags is None else wandb_tags
            if "Condition Prediction" not in wandb_tags:
                wandb_tags.append("Condition Prediction")
            wandb_run = wandb.init(  # type: ignore
                project=wandb_project,
                entity=wandb_entity,
                tags=wandb_tags,
                group=wandb_group,
            )
            callbacks.extend(
                [
                    WandbMetricsLogger(),
                    WandbModelCheckpoint("models", save_best_only=True),
                ]
            )

        use_multiprocessing = True if workers > 0 else False
        h = model.fit(
            train_generator,
            epochs=epochs,
            verbose=1,
            validation_data=val_generator,
            callbacks=callbacks,
            use_multiprocessing=use_multiprocessing,
            workers=workers,
        )
        update_teacher_forcing_model_weights(
            update_model=pred_model, to_copy_model=model
        )

        # Save the train and val metrics
        train_val_file_path = output_folder_path / "train_val_metrics.json"
        train_val_metrics_dict = h.history
        train_val_metrics_dict["trust_labelling"] = trust_labelling
        with open(train_val_file_path, "w") as file:
            json.dump(train_val_metrics_dict, file)

        # TODO: Save the model
        # model_save_file_path = output_folder_path / "models"
        # model.save(model_save_file_path)

        ### Evaluation ####
        post_training_plots(
            h, output_folder_path=output_folder_path, molecule_columns=molecule_columns
        )

        # Save the final performance on the test set
        if evaluate_on_test_data:
            # Evaluate the model on the test set
            test_metrics = model.evaluate(
                test_generator,
                use_multiprocessing=use_multiprocessing,
                workers=workers,
            )
            test_metrics_dict = dict(zip(model.metrics_names, test_metrics))
            test_metrics_dict["trust_labelling"] = trust_labelling

            ### Grouped scores
            predictions = model.predict(
                test_generator,
                use_multiprocessing=use_multiprocessing,
                workers=workers,
            )

            # Solvent scores
            solvent_scores = get_grouped_scores(
                y_test_data[:2], predictions[:2], encoders[:2]
            )
            test_metrics_dict["solvent_accuracy"] = np.mean(solvent_scores)

            # 3 agents scores
            agent_scores = get_grouped_scores(
                y_test_data[2:], predictions[2:], encoders[2:]
            )
            test_metrics_dict["three_agents_accuray"] = np.mean(agent_scores)

            # Overall scores
            overall_scores = np.stack([solvent_scores, agent_scores], axis=1).all(
                axis=1
            )
            test_metrics_dict["overall_accuracy"] = np.mean(overall_scores)

            # Save the test metrics
            test_metrics_file_path = output_folder_path / "test_metrics.json"
            # Save the dictionary as a JSON file
            with open(test_metrics_file_path, "w") as file:
                json.dump(test_metrics_dict, file)

            if wandb_logging:
                artifact = wandb.Artifact(  # type: ignore
                    name="test_metrics",
                    type="metrics",
                    description="Metrics on the test set",
                )
                artifact.add_dir(output_folder_path)
                wandb_run.log_artifact(artifact)  # type: ignore


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
    "--generate_fingerprints",
    default=False,
    type=bool,
    show_default=True,
    help="If True, will generate fingerprints on the fly instead of loading them from memory",
)
@click.option(
    "--workers",
    default=0,
    type=int,
    help="The number of workers to use for generating fingerprints. Defaults to 75\% of the CPUs on the machine. Defaults to 0",
)
@click.option(
    "--fp_size",
    default=2048,
    type=int,
    help="The size of the fingerprint used in fingerprint generation",
)
@click.option(
    "--dropout",
    default=0.2,
    type=float,
    help="The dropout rate used in the model",
)
@click.option(
    "--hidden_size_1",
    default=1024,
    type=int,
    help="The size of the first hidden layer in the model",
)
@click.option(
    "--hidden_size_2",
    default=100,
    type=int,
    help="The size of the second hidden layer in the model",
)
@click.option(
    "--lr",
    default=0.01,
    type=float,
    help="The learning rate used in the model",
)
@click.option(
    "--batch_size",
    default=512,
    type=int,
    help="The batch size used during training of the model",
)
@click.option(
    "--wandb_logging",
    default=True,
    type=bool,
    help="If True, will log to wandb",
)
@click.option(
    "--wandb_entity",
    default=None,
    type=str,
    help="The entity to use for logging to wandb",
)
@click.option(
    "--wandb_project",
    default="orderly",
    type=str,
    help="The project to use for logging to wandb",
)
@click.option(
    "--wandb_tags", multiple=True, default=None, help="Tags for weights and biases run"
)
@click.option(
    "--wandb_group",
    default=None,
    type=str,
    help="The group to use for logging to wandb",
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
    generate_fingerprints: bool,
    workers: int,
    fp_size: int,
    dropout: float,
    hidden_size_1: int,
    hidden_size_2: int,
    lr: float,
    batch_size: int,
    wandb_logging: bool,
    wandb_project: str,
    wandb_entity: Optional[str],
    wandb_tags: List[str],
    wandb_group: Optional[str],
    overwrite: bool,
    log_file: pathlib.Path = pathlib.Path("model.log"),
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
    main(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        output_folder_path=output_folder_path,
        train_fraction=train_fraction,
        train_val_split=train_val_split,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        evaluate_on_test_data=evaluate_on_test_data,
        generate_fingerprints=generate_fingerprints,
        workers=workers,
        fp_size=fp_size,
        dropout=dropout,
        hidden_size_1=hidden_size_1,
        hidden_size_2=hidden_size_2,
        lr=lr,
        batch_size=batch_size,
        wandb_logging=wandb_logging,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_tags=list(wandb_tags),
        wandb_group=wandb_group,
        overwrite=overwrite,
        log_file=log_file,
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
    generate_fingerprints: bool,
    workers: int,
    fp_size: int,
    dropout: float,
    hidden_size_1: int,
    hidden_size_2: int,
    lr: float,
    batch_size: int,
    wandb_logging: bool,
    wandb_project: str,
    wandb_entity: Optional[str],
    wandb_tags: List[str],
    wandb_group: Optional[str],
    overwrite: bool,
    log_file: pathlib.Path = pathlib.Path("model.log"),
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
    train_data_path = pathlib.Path(train_data_path)
    test_data_path = pathlib.Path(test_data_path)
    output_folder_path = pathlib.Path(output_folder_path)

    log_file = pathlib.Path(output_folder_path) / f"model.log"
    if log_file != "default_path_model.log":
        log_file = pathlib.Path(log_file)

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
        generate_fingerprints=generate_fingerprints,
        fp_size=fp_size,
        dropout=dropout,
        hidden_size_1=hidden_size_1,
        hidden_size_2=hidden_size_2,
        lr=lr,
        batch_size=batch_size,
        workers=workers,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        evaluate_on_test_data=evaluate_on_test_data,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_logging=wandb_logging,
        wandb_tags=list(wandb_tags),
        wandb_group=wandb_group,
    )

    instance.run_model_arguments()

    LOG.info(f"Completed model training, saving to {output_folder_path}")

    end_time = datetime.datetime.now()
    LOG.info("Training complete, duration: {}".format(end_time - start_time))
