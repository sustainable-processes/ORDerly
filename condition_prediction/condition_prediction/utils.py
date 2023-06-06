import math
import os
import socket
from collections import Counter
from copy import deepcopy
from datetime import datetime
from datetime import datetime as dt
from itertools import product
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from keras import callbacks
from sklearn.preprocessing import OneHotEncoder
from wandb.sdk.wandb_run import Run


def log_dir(prefix="", comment=""):
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(
        "runs", prefix + current_time + "_" + socket.gethostname() + comment
    )
    return log_dir


def apply_train_ohe_fit(df, train_idx, val_idx, test_idx=None, tensor_func=None):
    """
    Apply one-hot encoding to a column of a dataframe, using the training data to fit the encoder.

    Args:
        df: dataframe containing the column to be one-hot encoded
        train_idx: indices of the training data
        val_idx: indices of the validation data
        test_idx: indices of the test data
        tensor_func: function to convert the output to a tensor

    """

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    _ = enc.fit(df.iloc[train_idx])

    encoded_names = set(df.iloc[train_idx][df.columns[0]])
    # When there's a value in `val_idx` or `test_idx` that is not in `encoded_names`, set it to "other", rather than simply having an OHE row of all 0s
    if "other" in encoded_names:
        val_idx_values = df.iloc[val_idx][df.columns[0]]
        test_idx_values = df.iloc[test_idx][df.columns[0]]
        # Set values in `val_idx` that are not in `encoded_names` to "other"
        df.iloc[val_idx, df.columns.get_loc(df.columns[0])] = val_idx_values.apply(
            lambda x: x if x in encoded_names else "other"
        )
        # Set values in `test_idx` that are not in `encoded_names` to "other"
        df.iloc[test_idx, df.columns.get_loc(df.columns[0])] = test_idx_values.apply(
            lambda x: x if x in encoded_names else "other"
        )

    _ohe = enc.transform(df)
    # _tr, _val = _ohe.iloc[train_idx].values, _ohe.iloc[val_idx].values
    _tr, _val = _ohe[train_idx], _ohe[val_idx]
    _tr, _val = _tr.astype("float32"), _val.astype("float32")
    if tensor_func is not None:
        _tr, _val = tensor_func(_tr), tensor_func(_val)

    if test_idx is not None:
        # _test = _ohe.iloc[test_idx].values
        _test = _ohe[test_idx]
        _test = _test.astype("float32")
        if tensor_func is not None:
            _test = tensor_func(_test)

    return _tr, _val, _test, enc


def get_grouped_scores(y_true, y_pred, encoders=None):
    """
    Get the accuracy of the predictions for a group of components (e.g. solvents or agents)

    """
    components_true = []
    if encoders is not None:
        for enc, components in zip(encoders, y_true):
            components_true.append(enc.inverse_transform(components))
        components_true = np.concatenate(components_true, axis=1)

        components_pred = []
        for enc, components in zip(encoders, y_pred):
            selection_idx = np.argmax(components, axis=1)
            one_hot_targets = np.eye(components.shape[1])[selection_idx]
            components_pred.append(enc.inverse_transform(one_hot_targets))
        components_pred = np.concatenate(components_pred, axis=1)
        # Inverse transform will return None for an unknown label
        # This will introduce None, where we should only have 'NULL'
    else:
        components_true = y_true
        components_pred = y_pred

    components_true = np.where(components_true == None, "NULL", components_true)
    components_pred = np.where(components_pred == None, "NULL", components_pred)

    sorted_arr1 = np.sort(components_true, axis=1)
    sorted_arr2 = np.sort(components_pred, axis=1)

    return np.equal(sorted_arr1, sorted_arr2).all(axis=1)


def get_grouped_scores_top_n(y_true, y_pred, encoders, top_n):
    """
    Get the top-n accuracy of the predictions for a group of components (e.g. solvents or agents)
    Aka beam search accuracy with beam size n (no pruning)

    Selecting the top 1 combination is easy: just pick the highests probability for each component
    However, predicting the top n combinations is more difficult, as we need to consider all possible combinations; e.g. would it be better to have the second best solvent1 and the best agent1, or the best solvent1 and the second best agent1?
    """
    assert len(encoders) == len(y_true) == len(y_pred)
    assert y_true[0].shape == y_pred[0].shape
    assert len(y_true) == len(encoders) == len(y_pred)
    num_components = len(encoders)

    components_true = []

    # Transform the one-hot encoding of the groun truth (y_true) back to their original labels (e.g. ["Water", "DMSO"])
    for enc, components in zip(encoders, y_true):
        components_true.append(enc.inverse_transform(components))
    components_true = np.concatenate(components_true, axis=1)

    # Components_pred will be an array of dimension b x n x d
    # where b is batch size (number of reactions)
    # n is the beam width, i.e. top n samples
    # d is the number of components being predicted (i.e. solvents, agents)

    components_pred = []
    prob_components_pred = []
    for enc, components in zip(encoders, y_pred):
        # Here we get the indices of top n predictions for each component
        selection_idx = np.argsort(components, axis=1)[:, -top_n:]
        # And the probability of those n selections
        prob_selection = np.sort(components, axis=1)[:, -top_n:]
        # And convert that to one-hot of shape (n_reactions, n_components{i.e. n}, n_classes)
        one_hot_targets = np.array(
            [np.eye(components.shape[1])[idx] for idx in selection_idx]
        )
        # And then we get the actual component names
        component_pred = [enc.inverse_transform(one_hot) for one_hot in one_hot_targets]

        component_pred = np.stack(component_pred, axis=0)
        components_pred.append(component_pred)
        prob_components_pred.append(prob_selection)
    components_pred = np.concatenate(components_pred, axis=2)
    prob_components_pred = np.array(prob_components_pred)
    prob_components_pred = np.transpose(prob_components_pred, (1, 2, 0))

    components_true = np.where(components_true == None, "NULL", components_true)
    components_pred = np.where(components_pred == None, "NULL", components_pred)

    assert components_pred.shape == prob_components_pred.shape
    # Now we need to find all the possible combinations of the top-n predictions, as well as the associated probabilities for each combination

    # Initialize an empty list to store the combinations
    all_combinations = []
    # Initialize an empty list to store the scores
    all_scores = []

    # Loop through all the elements in a batch
    for i in range(components_pred.shape[0]):
        # Initialize an empty list for the current elements
        current_combinations = []
        current_scores = []
        # Get all combinations of the top-n predictions
        for combo in product(range(top_n), repeat=num_components):
            # Extract the combination
            current_combo = [
                components_pred[i][combo[j]][j] for j in range(num_components)
            ]
            current_score = math.prod(
                prob_components_pred[i][combo[j]][j] for j in range(num_components)
            )
            # Append it to the current list
            current_combinations.append(current_combo)
            current_scores.append(current_score)
        # Append the current combinations to the all combinations
        all_combinations.append(current_combinations)
        all_scores.append(current_scores)

    # Convert the list to a numpy array
    all_combinations = np.array(all_combinations)
    all_scores = np.array(all_scores)

    # Rank the combinations based on the scores
    ranking_idx = np.argsort(all_scores, axis=1)[:, ::-1]
    ranked_components_pred = np.take_along_axis(
        all_combinations, ranking_idx[..., None], axis=1
    )

    # Finally, simply slice the tensor to get the top n combinations
    ranked_components_pred = ranked_components_pred[:, :top_n, :]

    # Checking if true component is within the top-n predicted components
    match = np.array(
        [
            any(
                np.array_equal(np.sort(true_mol_combo), np.sort(mol_combo_suggestions))
                for mol_combo_suggestions in ranked_components_pred[i]
            )
            for i, true_mol_combo in enumerate(components_true)
        ]
    )

    return match


def frequency_informed_accuracy(data_train, data_test, top_n: int = 1):
    """
    Choose the most frequent combination of components in the training data and use that to predict the combination in the test data.

    TODO: This works, but there MUST be a way to do it more efficiently...
    """
    data_train_np = np.array(data_train).transpose()
    data_test_np = np.array(data_test).transpose()
    data_train_np = np.where(data_train_np == None, "NULL", data_train_np)
    data_test_np = np.where(data_test_np == None, "NULL", data_test_np)
    data_train_np = np.sort(data_train_np, axis=1)
    data_test_np = np.sort(data_test_np, axis=1)

    data_train_list = [tuple(row) for row in data_train_np]
    data_test_list = [tuple(row) for row in data_test_np]

    row_counts = Counter(data_train_list)

    # Find the most frequent row and its count
    most_frequent_rows = row_counts.most_common(top_n)

    # Count the occurrences of the most frequent row in data_train_np
    correct_predictions = 0
    for row in most_frequent_rows:
        correct_predictions += data_test_list.count(row[0])

    return correct_predictions / len(data_test_list), most_frequent_rows


def get_random_splits(n_indices, train_fraction, train_val_split, random_seed=54321):
    # Get indices for train and val
    rng = np.random.default_rng(random_seed)
    train_val_indexes = np.arange(n_indices)
    rng.shuffle(train_val_indexes)
    train_idx = train_val_indexes[
        : int(train_val_indexes.shape[0] * train_val_split * train_fraction)
    ]
    val_idx = train_val_indexes[int(train_val_indexes.shape[0] * train_val_split) :]
    return train_idx, val_idx


def post_training_plots(h, output_folder_path, molecule_columns: List[str]):
    mol_1_col = molecule_columns[0]
    mol_2_col = molecule_columns[1]
    mol_3_col = molecule_columns[2]
    mol_4_col = molecule_columns[3]
    mol_5_col = molecule_columns[4]

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

    # Save the train_val_loss plot
    plt.plot(h.history["loss"], label="loss")
    plt.plot(h.history["val_loss"], label="val_loss")
    plt.legend()
    output_file_path = output_folder_path / "train_val_loss.png"
    plt.savefig(output_file_path, bbox_inches="tight", dpi=600)


class TrainingMetrics(callbacks.Callback):
    def __init__(self, n_train: int, batch_size: int):
        super().__init__()
        self.n_train = n_train
        self.batch_size = batch_size

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = dt.now()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = (dt.now() - self.start_time).total_seconds()
        training_throughput = self.n_train / epoch_time
        logs.update(
            {
                "training_throughput": training_throughput,
                "time_per_step": self.batch_size / training_throughput,
            }
        )


def jsonify_dict(d, copy=True):
    """Make dictionary JSON serializable"""
    if copy:
        d = deepcopy(d)
    for k, v in d.items():
        if type(v) == np.ndarray:
            d[k] = v.tolist()
        elif type(v) == list:
            d[k] = jsonify_list(v)
        elif type(v) == dict:
            d[k] = jsonify_dict(v)
        elif type(v) in (np.int64, np.int32, np.int8):
            d[k] = int(v)
        elif type(v) in (np.float16, np.float32, np.float64):
            d[k] = float(v)
        elif type(v) in [str, int, float, bool, tuple] or v is None:
            pass
        else:
            raise TypeError(
                f"Cannot jsonify type for key ({k}) with value {v} and type {type(v)}."
            )
    return d


def unjsonify_dict(d, copy=True):
    """Convert JSON back to proper types"""
    if copy:
        d = deepcopy(d)
    for k, v in d.items():
        if type(v) == list:
            d[k] = listtonumpy(v)
        elif type(v) == dict:
            d[k] = unjsonify_dict(v)
        elif type(v) in [str, int, float, bool, tuple] or v is None:
            pass
        else:
            raise TypeError(
                f"Cannot unjsonify type for key ({k}) with value {v} and type {type(v)}."
            )
    return d


def jsonify_list(a, copy=True):
    if copy:
        a = deepcopy(a)
    for i, l in enumerate(a):
        if type(l) == list:
            a[i] = jsonify_list(l)
        elif type(l) == dict:
            a[i] = jsonify_dict(l)
        elif type(l) == np.ndarray:
            a[i] = l.tolist()
        elif type(l) in (np.float16, np.float32, np.float64):
            a[i] = float(l)
        elif type(l) in [str, int, float, bool, tuple] or l is None:
            pass
        else:
            raise TypeError(
                f"Cannot jsonify type for index ({i}) with value {l} and type {type(l)}."
            )
    return a


def listtonumpy(a, copy=True):
    if copy:
        a = deepcopy(a)
    transform_all = True
    for i, l in enumerate(a):
        if type(l) == dict:
            a[i] = unjsonify_dict(l)
            transform_all = False
        elif type(l) == list:
            a[i] = listtonumpy(l)
            transform_all = False
        elif type(l) in [str, float, bool, int] or l is None:
            pass
        elif type(l) == tuple:
            transform_all = False
        else:
            raise TypeError(f"Cannot jsonify type for {l}: {type(l)}.")
    if transform_all:
        a = np.array(a)
    return a


def download_model_from_wandb(
    run: Run, alias="best", weights_file="weights.best.hdf5", root=None
) -> Tuple[Path, str]:
    """Download best model from wandb

    Arguments
    ----------
    run: wandb.Run


    Returns
    -------
    Path to best model checkpoint

    """
    # Get best checkpoint
    artifacts = run.logged_artifacts()  # type: ignore
    ckpt_path = None
    artifact_name = None
    for artifact in artifacts:
        if artifact.type == "model" and alias in artifact.aliases:
            ckpt_path = artifact.download(root=root)
            artifact_name = artifact.name
    if ckpt_path is None or artifact_name is None:
        raise ValueError("No checkpoint found with alias: {}".format(alias))
    ckpt_path = Path(ckpt_path) / weights_file

    return ckpt_path, artifact_name
