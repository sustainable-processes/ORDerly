import pytest


def test_hello_world():
    assert True


def test_molecule_names_not_empty():
    from orderly.data.test_data import get_path_of_molecule_names
    import pandas as pd
    import os

    molecule_names_folder_path = get_path_of_molecule_names()

    files = os.listdir(molecule_names_folder_path)
    for file in files:
        if file.endswith(".parquet") or file.endswith(".pkl"):
            file_path = os.path.join(molecule_names_folder_path, file)
            df = pd.read_parquet(file_path)
            assert not df.empty


def check_frequency_of_occurance(
    series,
    column_name,
    min_frequency_of_occurrence,
    include_other_category,
    map_rare_to_other_threshold,
):
    # series could be df['agent_0'], df['reagent_1'], df['solvent_0'], etc.
    item_frequencies = series[series != "other"].value_counts()

    # Check that the item with the lowest frequency appears at least `min_frequency_of_occurrence` times
    if len(item_frequencies) > 0:
        least_common_frequency = item_frequencies.iloc[-1]
        if include_other_category:
            # If 'other' is included, then the least common item must appear at least `min_frequency_of_occurrence` times
            assert (
                least_common_frequency >= map_rare_to_other_threshold
            ), f"Error in frequencies of {column_name}"
        else:
            # If 'other' is not included, then the least common item must appear at least `min_frequency_of_occurrence` times
            assert (
                least_common_frequency >= min_frequency_of_occurrence
            ), f"Error in frequencies of {column_name}"
    else:
        # If there are no items other than 'other', the test passes
        pass


def get_cleaned_df(
    trust_labelling,
    output_path,
    consistent_yield,
    num_reactant,
    num_product,
    num_solv,
    num_agent,
    num_cat,
    num_reag,
    min_frequency_of_occurance_primary,
    min_frequency_of_occurance_secondary,
    include_other_category,
    map_rare_to_other_threshold,
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
        min_frequency_of_occurance_primary=min_frequency_of_occurance_primary,
        min_frequency_of_occurance_secondary=min_frequency_of_occurance_secondary,
        include_other_category=include_other_category,
        map_rare_to_other_threshold=map_rare_to_other_threshold,
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
            [True, True, 5, 5, 2, 3, 0, 0, 15, 15, True, 3],
            id="trust_labelling:T|consistent_yield:T|include_other_category:T",
        ),
        pytest.param(
            [True, False, 5, 5, 2, 3, 0, 0, 15, 15, True, 3],
            id="trust_labelling:T|consistent_yield:F|include_other_category:T",
        ),
        pytest.param(
            [True, False, 5, 5, 2, 3, 0, 0, 15, 15, True, 3],
            id="trust_labelling:T|consistent_yield:T|include_other_category:F",
        ),
        pytest.param(
            [True, False, 5, 5, 2, 3, 0, 0, 15, 15, False, 3],
            id="trust_labelling:T|consistent_yield:F|include_other_category:F",
        ),
        pytest.param(
            [False, True, 5, 5, 2, 3, 0, 0, 15, 15, True, 3],
            id="trust_labelling:F|consistent_yield:T|include_other_category:T",
        ),
        pytest.param(
            [False, False, 5, 5, 2, 3, 0, 0, 15, 15, True, 3],
            id="trust_labelling:F|consistent_yield:F|include_other_category:T",
        ),
        pytest.param(
            [False, False, 5, 5, 2, 3, 0, 0, 15, 15, True, 3],
            id="trust_labelling:F|consistent_yield:T|include_other_category:F",
        ),
        pytest.param(
            [False, False, 5, 5, 2, 3, 0, 0, 15, 15, False, 3],
            id="trust_labelling:F|consistent_yield:F|include_other_category:F",
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
            [True, True, 5, 5, 2, 3, 0, 0, 15, 15, True, 3],
            id="trust_labelling:T|consistent_yield:T|include_other_category:T",
        ),
        pytest.param(
            [True, False, 5, 5, 2, 3, 0, 0, 15, 15, True, 3],
            id="trust_labelling:T|consistent_yield:F|include_other_category:T",
        ),
        pytest.param(
            [True, False, 5, 5, 2, 3, 0, 0, 15, 15, True, 3],
            id="trust_labelling:T|consistent_yield:T|include_other_category:F",
        ),
        pytest.param(
            [True, False, 5, 5, 2, 3, 0, 0, 15, 15, False, 3],
            id="trust_labelling:T|consistent_yield:F|include_other_category:F",
        ),
        pytest.param(
            [True, False, 5, 5, 5, 5, 5, 5, 15, 15, True, 5],
            marks=pytest.mark.xfail(
                reason="AssertionError: Invalid input: If trust_labelling=True in orderly.extract, then num_cat and num_reag must be 0. If trust_labelling=False, then num_agent must be 0."
            ),
            id="trust_labelling:T|consistent_yield:T|include_other_category:F|fives",
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
            [True, True, 5, 5, 2, 3, 0, 0, 15, 15, True, 3],
            id="trust_labelling:T|consistent_yield:T|include_other_category:T",
        ),
        pytest.param(
            [True, False, 5, 5, 2, 3, 0, 0, 15, 15, True, 3],
            id="trust_labelling:T|consistent_yield:F|include_other_category:T",
        ),
        pytest.param(
            [True, False, 5, 5, 2, 3, 0, 0, 15, 15, True, 3],
            id="trust_labelling:T|consistent_yield:T|include_other_category:F",
        ),
        pytest.param(
            [True, False, 5, 5, 2, 3, 0, 0, 15, 15, False, 3],
            id="trust_labelling:T|consistent_yield:F|include_other_category:F",
        ),
        pytest.param(
            [True, False, 5, 5, 5, 5, 5, 5, 15, 15, True, 5],
            marks=pytest.mark.xfail(
                reason="AssertionError: Invalid input: If trust_labelling=True in orderly.extract, then num_cat and num_reag must be 0. If trust_labelling=False, then num_agent must be 0."
            ),
            id="trust_labelling:T|consistent_yield:T|include_other_category:F|fives",
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
        min_frequency_of_occurance_primary,
        min_frequency_of_occurance_secondary,
        include_other_category,
        map_rare_to_other_threshold,
    ) = params

    cols = cleaned_df.columns
    for col in cols:
        if col in ["agent_0", "solvent_0", "catalyst_0", "reagent_0"]:
            check_frequency_of_occurance(
                cleaned_df[col],
                col,
                min_frequency_of_occurance_primary,
                include_other_category,
                map_rare_to_other_threshold,
            )
        elif col in ["agent_1", "solvent_1", "catalyst_1", "reagent_1"]:
            check_frequency_of_occurance(
                cleaned_df[col],
                col,
                min_frequency_of_occurance_secondary,
                include_other_category,
                map_rare_to_other_threshold,
            )
