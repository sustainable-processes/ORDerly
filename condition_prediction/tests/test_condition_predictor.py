import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from condition_prediction.run import ConditionPrediction

# Generate 2 dataframes each with 2 columns that have 3 classes
# (optional) Use one hot encoder to generate one-hot encoded arryas
# Run through get_grouped_scores and check if you get the expected results


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
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
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
    scores = ConditionPrediction.get_grouped_scores(
        ground_truth, prediction, encoders=encoders
    )
    assert np.mean(scores) == 0.75
