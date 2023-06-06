import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from condition_prediction.utils import get_grouped_scores, get_grouped_scores_top_n

# Generate 2 dataframes each with 2 columns that have 3 classes
# (optional) Use one hot encoder to generate one-hot encoded arryas
# Run through get_grouped_scores and check if you get the expected results


def test_hello_world() -> None:
    assert True


@pytest.mark.parametrize(
    "one_hot_encode",
    [(True,), (False,)],
)
def test_grouped_scores(one_hot_encode: bool):
    ground_truth_df = pd.DataFrame(
        [["THF", "Water"], ["Ethanol", "THF"], ["Methanol", None], ["Water", None]],
        columns=["solvent_1", "solvent_2"],
    )
    prediction_df = pd.DataFrame(
        [
            ["Water", "THF"],
            ["Ethanol", "THF"],
            ["Methanol", "Water"],
            ["Water", None],
        ],
        columns=["solvent_1", "solvent_2"],
    )

    if one_hot_encode:
        # One hot encode the dataframes
        ground_truth = []
        prediction = []
        encoders = []
        for col in ground_truth_df.columns:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            ground_truth_col = encoder.fit_transform(ground_truth_df[[col]])
            prediction_col = encoder.transform(prediction_df[[col]])
            ground_truth.append(ground_truth_col)
            prediction.append(prediction_col)
            encoders.append(encoder)
    else:
        ground_truth = ground_truth_df.to_numpy()
        prediction = prediction_df.to_numpy()
        encoders = None

    # Average score should be 75%
    scores = get_grouped_scores(ground_truth, prediction, encoders=encoders)
    assert np.mean(scores) == 0.75


@pytest.mark.parametrize(
    "top_n, expected_accuracy",
    (
        [1, round(1 / 6, 2)],
        [2, round(2 / 6, 2)],
        [3, round(2 / 6, 2)],
        [4, round(3 / 6, 2)],
    ),
)
def test_get_grouped_scores_top_n(top_n: int, expected_accuracy: float):
    solvents_list_1 = ["THF", "Ethanol", "Methanol", "Water", "Furan"]
    solvents_list_2 = ["Water", "THF", "Methanol", None]
    solvents_lists = [solvents_list_1, solvents_list_2]

    ground_truth_df = pd.DataFrame(
        [
            ["THF", "Water"],
            ["Ethanol", "THF"],
            ["Methanol", None],
            ["Water", None],
            ["THF", "Methanol"],
            ["Furan", "THF"],
        ],
        columns=["solvent_1", "solvent_2"],
    )

    s1 = np.array(
        [
            [0.14, 0.31, 0.18, 0.25, 0.12],
            [0.32, 0.08, 0.19, 0.22, 0.19],
            [0.13, 0.21, 0.23, 0.25, 0.18],
            [0.23, 0.24, 0.26, 0.09, 0.18],
            [0.15, 0.22, 0.31, 0.18, 0.14],
            [0.12, 0.25, 0.16, 0.29, 0.18],
        ]
    )

    s2 = np.array(
        [
            [0.18, 0.37, 0.29, 0.16],
            [0.32, 0.13, 0.23, 0.32],
            [0.23, 0.29, 0.21, 0.27],
            [0.25, 0.22, 0.34, 0.19],
            [0.16, 0.37, 0.29, 0.18],
            [0.61, 0.12, 0.15, 0.11],
        ]
    )

    prediction_probability = [s1, s2]

    # One hot encode the dataframes
    ground_truth = []
    encoders = []
    for col, solvent_list in zip(ground_truth_df.columns, solvents_lists):
        encoder = OneHotEncoder(
            categories=[solvent_list], sparse_output=False, handle_unknown="ignore"
        )
        ground_truth_col = encoder.fit_transform(ground_truth_df[[col]])
        ground_truth.append(ground_truth_col)
        encoders.append(encoder)

    assert ground_truth[0].shape == s1.shape
    assert ground_truth[1].shape == s2.shape

    scores = get_grouped_scores_top_n(
        ground_truth, prediction_probability, encoders=encoders, top_n=top_n
    )

    assert round(np.mean(scores), 2) == expected_accuracy

    if top_n == 1:
        alt_scores = get_grouped_scores(
            ground_truth, prediction_probability, encoders=encoders
        )
        assert np.array_equal(alt_scores, scores)
