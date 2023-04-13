import pytest


def test_hello_world():
    assert True


def test_molecule_names_not_empty():
    from orderly.data.test_data import get_path_of_molecule_names
    import pandas as pd
    import os

    all_empty = True

    molecule_names_folder_path = get_path_of_molecule_names()

    files = os.listdir(molecule_names_folder_path)
    for file in files:
        if file.endswith(".pkl"):  # or file.endswith(".parquet")
            file_path = os.path.join(molecule_names_folder_path, file)
            # df = pd.read_parquet(file_path)
            # if not df.empty:
            #     all_empty = False
            molecule_names_list = pd.read_pickle(file_path)
            if len(molecule_names_list) > 0:
                all_empty = False
    assert not all_empty


@pytest.fixture
def toy_dict():
    toy_dict = {
        "reactant_0": ["B", "A", "F", "A"],
        "reactant_1": ["D", "A", "G", "B"],
        "product_0": ["C", "A", "E", "A"],
        "product_1": ["E", "G", "C", "H"],
        "agent_0": ["D", "F", "D", "B"],
        "agent_1": ["C", "E", "G", "A"],
        "solvent_0": ["E", "B", "G", "C"],
        "solvent_1": ["C", "D", "B", "G"],
        "solvent_2": ["D", "B", "F", "G"],
    }

    return toy_dict


@pytest.mark.parametrize(
    "columns_to_count_from, expected_total_value_counts",
    (
        pytest.param(
            [
                "reactant_0",
                "reactant_1",
                "product_0",
                "product_1",
                "agent_0",
                "agent_1",
                "solvent_0",
                "solvent_1",
                "solvent_2",
            ],
            {"A": 6, "B": 6, "C": 5, "D": 5, "E": 4, "F": 3, "G": 6, "H": 1},
            id="all_columns",
        ),
        pytest.param(
            ["agent_0", "agent_1", "solvent_0", "solvent_1", "solvent_2"],
            {"A": 1, "B": 4, "C": 3, "D": 4, "E": 2, "F": 2, "G": 4},
            id="agent_and_solvent_columns",
        ),
    ),
)
def test_get_value_counts(
    toy_dict, columns_to_count_from, expected_total_value_counts
) -> None:
    import orderly.clean.cleaner
    import pandas as pd

    df = pd.DataFrame(toy_dict)

    total_value_counts = orderly.clean.cleaner.Cleaner._get_value_counts(
        df, columns_to_count_from
    )

    expected_total_value_counts = pd.Series(expected_total_value_counts).sort_values(
        ascending=False
    )

    assert total_value_counts.equals(
        pd.Series(expected_total_value_counts)
    ), f"Got: {total_value_counts}, expected: {expected_total_value_counts},"


@pytest.mark.parametrize(
    "columns_to_transform, value_counts_dict, min_frequency_of_occurrence, expected_dict,",
    (
        pytest.param(
            [
                "reactant_0",
                "reactant_1",
                "product_0",
                "product_1",
                "agent_0",
                "agent_1",
                "solvent_0",
                "solvent_1",
                "solvent_2",
            ],
            {"A": 6, "B": 6, "C": 5, "D": 5, "E": 4, "F": 3, "G": 6, "H": 1},
            4,
            {
                "reactant_0": ["B", "A", "other", "A"],
                "reactant_1": ["D", "A", "G", "B"],
                "product_0": ["C", "A", "E", "A"],
                "product_1": ["E", "G", "C", "other"],
                "agent_0": ["D", "other", "D", "B"],
                "agent_1": ["C", "E", "G", "A"],
                "solvent_0": ["E", "B", "G", "C"],
                "solvent_1": ["C", "D", "B", "G"],
                "solvent_2": ["D", "B", "other", "G"],
            },
            id="all_columns",
        ),
        pytest.param(
            ["agent_0", "agent_1", "solvent_0", "solvent_1", "solvent_2"],
            {"A": 1, "B": 4, "C": 3, "D": 4, "E": 2, "F": 2, "G": 4},
            3,
            {
                "reactant_0": ["B", "A", "F", "A"],
                "reactant_1": ["D", "A", "G", "B"],
                "product_0": ["C", "A", "E", "A"],
                "product_1": ["E", "G", "C", "H"],
                "agent_0": ["D", "other", "D", "B"],
                "agent_1": ["C", "other", "G", "other"],
                "solvent_0": ["other", "B", "G", "C"],
                "solvent_1": ["C", "D", "B", "G"],
                "solvent_2": ["D", "B", "other", "G"],
            },
            id="agent_and_solvent_columns",
        ),
    ),
)
def test_map_rare_molecules_to_other(
    toy_dict,
    columns_to_transform,
    value_counts_dict,
    min_frequency_of_occurrence,
    expected_dict,
) -> None:
    import pandas as pd
    import orderly.clean.cleaner

    df = pd.DataFrame(toy_dict)
    value_counts_series = pd.Series(value_counts_dict)

    df = orderly.clean.cleaner.Cleaner._map_rare_molecules_to_other(
        df, columns_to_transform, value_counts_series, min_frequency_of_occurrence
    )

    expected_df = pd.DataFrame(expected_dict)

    assert df.equals(expected_df), f"Got: {df}, expected: {expected_df},"


@pytest.mark.parametrize(
    "columns_to_transform, value_counts_dict, min_frequency_of_occurrence, expected_dict,",
    (
        pytest.param(
            [
                "reactant_0",
                "reactant_1",
                "product_0",
                "product_1",
                "agent_0",
                "agent_1",
                "solvent_0",
                "solvent_1",
                "solvent_2",
            ],
            {"A": 6, "B": 6, "C": 5, "D": 5, "E": 4, "F": 3, "G": 6, "H": 1},
            4,
            {
                "reactant_0": ["B"],
                "reactant_1": ["D"],
                "product_0": ["C"],
                "product_1": ["E"],
                "agent_0": ["D"],
                "agent_1": ["C"],
                "solvent_0": ["E"],
                "solvent_1": ["C"],
                "solvent_2": ["D"],
            },
            id="all_columns",
        ),
        pytest.param(
            ["agent_0", "agent_1", "solvent_0", "solvent_1", "solvent_2"],
            {"A": 1, "B": 4, "C": 3, "D": 4, "E": 2, "F": 2, "G": 4},
            2,
            {
                "reactant_0": ["B", "A", "F"],
                "reactant_1": ["D", "A", "G"],
                "product_0": ["C", "A", "E"],
                "product_1": ["E", "G", "C"],
                "agent_0": ["D", "F", "D"],
                "agent_1": ["C", "E", "G"],
                "solvent_0": ["E", "B", "G"],
                "solvent_1": ["C", "D", "B"],
                "solvent_2": ["D", "B", "F"],
            },
            id="agent_and_solvent_columns",
        ),
    ),
)
def test_remove_rare_molecules(
    toy_dict,
    columns_to_transform,
    value_counts_dict,
    min_frequency_of_occurrence,
    expected_dict,
) -> None:
    import pandas as pd
    import orderly.clean.cleaner

    df = pd.DataFrame(toy_dict)
    value_counts_series = pd.Series(value_counts_dict)

    df = orderly.clean.cleaner.Cleaner._remove_rare_molecules(
        df, columns_to_transform, value_counts_series, min_frequency_of_occurrence
    )

    expected_df = pd.DataFrame(expected_dict)

    assert df.equals(expected_df), f"Got: {df}, expected: {expected_df},"


def check_frequency_of_occurrence(
    df,
    min_frequency_of_occurrence,
):
    import pandas as pd

    # Define the list of columns to check
    columns_to_check = [
        col
        for col in df.columns
        if col.startswith(("agent", "solvent", "reagent", "catalyst"))
    ]

    # Initialize a list to store the results
    results = []

    # Loop through the columns
    for col in columns_to_check:
        # Get the value counts for the column
        results += [df[col].value_counts()]

    total_value_counts = pd.concat(results, axis=0, sort=True).groupby(level=0).sum()
    if "other" in total_value_counts.index:
        total_value_counts = total_value_counts.drop("other")
    total_value_counts = total_value_counts.sort_values(ascending=True)

    assert (
        total_value_counts.iloc[0] >= min_frequency_of_occurrence
    ), f"{min_frequency_of_occurrence=} is not being respected with {total_value_counts.iloc[0]} occurrences of {total_value_counts.index[0]}."


def get_cleaned_df(
    output_path,
    trust_labelling,
    consistent_yield,
    num_reactant,
    num_product,
    num_solv,
    num_agent,
    num_cat,
    num_reag,
    min_frequency_of_occurrence,
    map_rare_molecules_to_other,
):
    import orderly.clean.cleaner
    import orderly.data

    pickles_path = (
        orderly.data.get_path_of_test_extracted_ords(trust_labelling=trust_labelling)
        / "pickled_data"
    )
    molecules_to_remove_path = (
        orderly.data.get_path_of_test_extracted_ords(trust_labelling=trust_labelling)
        / "all_molecule_names.pkl"
    )

    orderly.clean.cleaner.main(
        clean_data_path=output_path / "orderly_ord.parquet",
        pickles_path=pickles_path,
        molecules_to_remove_path=molecules_to_remove_path,
        consistent_yield=consistent_yield,
        num_reactant=num_reactant,
        num_product=num_product,
        num_solv=num_solv,
        num_agent=num_agent,
        num_cat=num_cat,
        num_reag=num_reag,
        min_frequency_of_occurrence=min_frequency_of_occurrence,
        map_rare_molecules_to_other=map_rare_molecules_to_other,
        disable_tqdm=False,
    )

    import pandas as pd

    return pd.read_parquet(output_path / "orderly_ord.parquet")


@pytest.fixture
def cleaned_df_params(tmp_path, request):
    return get_cleaned_df(tmp_path, *request.param), request.param


@pytest.mark.parametrize(
    "cleaned_df_params",
    (
        pytest.param(
            [False, False, 5, 5, 2, 3, 0, 0, 15, False],
            id="trust_labelling:F|consistent_yield:F|map_rare_molecules_to_other:F",
        ),
        pytest.param(
            [True, False, 5, 5, 2, 0, 2, 1, 15, False],
            id="trust_labelling:T|consistent_yield:F|map_rare_molecules_to_other:F",
        ),
        pytest.param(
            [False, True, 5, 5, 2, 3, 0, 0, 15, False],
            id="trust_labelling:F|consistent_yield:T|map_rare_molecules_to_other:F",
        ),
        pytest.param(
            [False, False, 5, 5, 2, 3, 0, 0, 15, True],
            id="trust_labelling:F|consistent_yield:F|map_rare_molecules_to_other:T",
        ),
        pytest.param(
            [True, True, 5, 5, 2, 0, 2, 1, 15, False],
            id="trust_labelling:T|consistent_yield:T|map_rare_molecules_to_other:F",
        ),
        pytest.param(
            [True, False, 5, 5, 2, 0, 2, 1, 15, True],
            id="trust_labelling:T|consistent_yield:F|map_rare_molecules_to_other:T",
        ),
        pytest.param(
            [False, True, 5, 5, 2, 3, 0, 0, 15, True],
            id="trust_labelling:F|consistent_yield:T|map_rare_molecules_to_other:T",
        ),
        pytest.param(
            [True, True, 5, 5, 2, 0, 2, 1, 15, True],
            id="trust_labelling:T|consistent_yield:T|map_rare_molecules_to_other:T",
        ),
    ),
    indirect=True,
)
def test_get_cleaned_df(cleaned_df_params):
    cleaned_df, params = cleaned_df_params
    assert not cleaned_df.empty


@pytest.mark.parametrize(
    "cleaned_df_params",
    (
        pytest.param(
            [False, False, 5, 5, 2, 3, 0, 0, 15, False],
            id="trust_labelling:F|consistent_yield:F|map_rare_molecules_to_other:F",
        ),
        pytest.param(
            [True, False, 5, 5, 2, 0, 2, 1, 15, False],
            id="trust_labelling:T|consistent_yield:F|map_rare_molecules_to_other:F",
        ),
        pytest.param(
            [False, True, 5, 5, 2, 3, 0, 0, 15, False],
            id="trust_labelling:F|consistent_yield:T|map_rare_molecules_to_other:F",
        ),
        pytest.param(
            [False, False, 5, 5, 2, 3, 0, 0, 15, True],
            id="trust_labelling:F|consistent_yield:F|map_rare_molecules_to_other:T",
        ),
        pytest.param(
            [True, True, 5, 5, 2, 0, 2, 1, 15, False],
            id="trust_labelling:T|consistent_yield:T|map_rare_molecules_to_other:F",
        ),
        pytest.param(
            [True, False, 5, 5, 2, 0, 2, 1, 15, True],
            id="trust_labelling:T|consistent_yield:F|map_rare_molecules_to_other:T",
        ),
        pytest.param(
            [False, True, 5, 5, 2, 3, 0, 0, 15, True],
            id="trust_labelling:F|consistent_yield:T|map_rare_molecules_to_other:T",
        ),
        pytest.param(
            [True, True, 5, 5, 2, 0, 2, 1, 15, True],
            id="trust_labelling:T|consistent_yield:T|map_rare_molecules_to_other:T",
        ),
        # XFAILS
        pytest.param(
            [False, True, 5, 5, 5, 5, 5, 5, 5, True],
            marks=pytest.mark.xfail(
                reason="AssertionError: Invalid input: If trust_labelling=True in orderly.extract, then num_cat and num_reag must be 0. If trust_labelling=False, then num_agent must be 0."
            ),
            id="trust_labelling:F|consistent_yield:T|map_rare_molecules_to_other:F|fives",
        ),
    ),
    indirect=True,
)
def test_number_of_columns(cleaned_df_params):
    cleaned_df, params = cleaned_df_params

    (
        _,
        _,
        num_reactant,
        num_product,
        num_solv,
        num_agent,
        num_cat,
        num_reag,
        _,
        _,
    ) = params

    # check that the number of columns is correct
    num_reactant_cols = 0
    num_product_cols = 0
    num_agent_cols = 0
    num_cat_cols = 0
    num_reag_cols = 0
    num_solv_cols = 0

    cols = cleaned_df.columns
    for col in cols:
        if col.startswith("react"):
            num_reactant_cols += 1
        elif col.startswith("prod"):
            num_product_cols += 1
        elif col.startswith("agent"):
            num_agent_cols += 1
        elif col.startswith("cat"):
            num_cat_cols += 1
        elif col.startswith("reag"):
            num_reag_cols += 1
        elif col.startswith("solv"):
            num_solv_cols += 1

    assert num_reactant_cols == num_reactant
    assert num_product_cols == num_product
    assert num_agent_cols == num_agent
    assert num_cat_cols == num_cat
    assert num_reag_cols == num_reag
    assert num_solv_cols == num_solv


@pytest.mark.parametrize(
    "cleaned_df_params",
    (
        pytest.param(
            [False, False, 5, 5, 2, 3, 0, 0, 15, False],
            id="trust_labelling:F|consistent_yield:F|map_rare_molecules_to_other:F",
        ),
        pytest.param(
            [True, False, 5, 5, 2, 0, 2, 1, 15, False],
            id="trust_labelling:T|consistent_yield:F|map_rare_molecules_to_other:F",
        ),
        pytest.param(
            [False, True, 5, 5, 2, 3, 0, 0, 15, False],
            id="trust_labelling:F|consistent_yield:T|map_rare_molecules_to_other:F",
        ),
        pytest.param(
            [False, False, 5, 5, 2, 3, 0, 0, 15, True],
            id="trust_labelling:F|consistent_yield:F|map_rare_molecules_to_other:T",
        ),
        pytest.param(
            [True, True, 5, 5, 2, 0, 2, 1, 15, False],
            id="trust_labelling:T|consistent_yield:T|map_rare_molecules_to_other:F",
        ),
        pytest.param(
            [True, False, 5, 5, 2, 0, 2, 1, 15, True],
            id="trust_labelling:T|consistent_yield:F|map_rare_molecules_to_other:T",
        ),
        pytest.param(
            [False, True, 5, 5, 2, 3, 0, 0, 15, True],
            id="trust_labelling:F|consistent_yield:T|map_rare_molecules_to_other:T",
        ),
        pytest.param(
            [True, True, 5, 5, 2, 0, 2, 1, 15, True],
            id="trust_labelling:T|consistent_yield:T|map_rare_molecules_to_other:T",
        ),
        # XFAILS
        pytest.param(
            [False, True, 5, 5, 5, 5, 5, 5, 5, True],
            marks=pytest.mark.xfail(
                reason="AssertionError: Invalid input: If trust_labelling=True in orderly.extract, then num_cat and num_reag must be 0. If trust_labelling=False, then num_agent must be 0."
            ),
            id="trust_labelling:F|consistent_yield:T|map_rare_molecules_to_other:F|fives",
        ),
    ),
    indirect=True,
)
def test_frequency(cleaned_df_params):
    cleaned_df, params = cleaned_df_params

    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        min_frequency_of_occurrence,
        _,
    ) = params

    check_frequency_of_occurrence(cleaned_df, min_frequency_of_occurrence)
