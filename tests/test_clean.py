from typing import List, Dict, Tuple, Any
import pathlib
import pytest


import pandas as pd


def test_hello_world() -> None:
    assert True


@pytest.fixture
def toy_dict() -> Dict[str, List[str]]:
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


def get_cleaned_df(
    output_path: pathlib.Path,
    trust_labelling: bool,
    consistent_yield: bool,
    num_reactant: int,
    num_product: int,
    num_solv: int,
    num_agent: int,
    num_cat: int,
    num_reag: int,
    min_frequency_of_occurrence: int,
    map_rare_molecules_to_other: bool,
    remove_with_unresolved_names: bool,
    replace_empty_with_none: bool,
    drop_duplicates: bool,
) -> pd.DataFrame:
    import orderly.clean.cleaner
    import orderly.data.test_data

    ord_extraction_path = (
        orderly.data.test_data.get_path_of_test_extracted_ords(
            trust_labelling=trust_labelling
        )
        / "extracted_ords"
    )
    molecules_to_remove_path = (
        orderly.data.test_data.get_path_of_test_extracted_ords(
            trust_labelling=trust_labelling
        )
        / "all_molecule_names.csv"
    )

    orderly.clean.cleaner.main(
        clean_data_path=output_path / "orderly_ord.parquet",
        ord_extraction_path=ord_extraction_path,
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
        remove_with_unresolved_names=remove_with_unresolved_names,
        replace_empty_with_none=replace_empty_with_none,
        drop_duplicates=drop_duplicates,
        disable_tqdm=False,
        overwrite=False,
    )

    import pandas as pd

    return pd.read_parquet(output_path / "orderly_ord.parquet")


@pytest.fixture
def cleaned_df_params(
    tmp_path: pathlib.Path, request: pytest.FixtureRequest
) -> Tuple[pd.DataFrame, List[Any]]:
    assert len(request.param) == 10
    updated_args = request.param + [
        True,
        True,
        True,
    ]  # remove_with_unresolved_names, replace_empty_with_none, drop_duplicates
    return (
        get_cleaned_df(
            tmp_path / "cleaned_df",
            *updated_args,
        ),
        request.param,
    )


@pytest.fixture
def cleaned_df_params_without_unresolved_names_and_duplicates(
    tmp_path: pathlib.Path, request: pytest.FixtureRequest
) -> Tuple[pd.DataFrame, List[Any]]:
    assert len(request.param) == 10
    updated_args = request.param + [
        False,
        True,
        False,
    ]  # remove_with_unresolved_names, replace_empty_with_none, drop_duplicates
    return (
        get_cleaned_df(
            tmp_path / "cleaned_df_params_without_unresolved_names_and_duplicates",
            *updated_args,
        ),
        request.param,
    )


@pytest.fixture
def cleaned_df_params_without_min_freq(
    tmp_path: pathlib.Path, request: pytest.FixtureRequest
) -> Tuple[pd.DataFrame, List[Any]]:
    import copy

    args = copy.copy(request.param)
    args[-2] = 0
    assert len(request.param) == 10
    updated_args = args + [
        True,
        True,
        True,
    ]  # remove_with_unresolved_names, replace_empty_with_none, drop_duplicates
    return (
        get_cleaned_df(
            tmp_path / "cleaned_df_params_without_min_freq",
            *updated_args,
        ),
        args,
    )


@pytest.fixture
def cleaned_df_params_without_min_freq_without_unresolved_names_and_duplicates(
    tmp_path: pathlib.Path, request: pytest.FixtureRequest
) -> Tuple[pd.DataFrame, List[Any]]:
    import copy

    args = copy.copy(request.param)
    args[-2] = 0
    assert len(request.param) == 10
    updated_args = args + [
        False,
        True,
        False,
    ]  # remove_with_unresolved_names, replace_empty_with_none, drop_duplicates
    return (
        get_cleaned_df(
            tmp_path
            / "cleaned_df_params_without_min_freq_without_unresolved_names_and_duplicates",
            *updated_args,
        ),
        args,
    )


def test_molecule_names_not_empty() -> None:
    import os
    import pathlib
    import pandas as pd
    import orderly.data.util

    from orderly.data.test_data import get_path_of_molecule_names

    all_empty = True

    molecule_names_folder_path = get_path_of_molecule_names()

    files = os.listdir(molecule_names_folder_path)
    for file in files:
        if file.endswith(".csv"):  # or file.endswith(".parquet")
            file_path = pathlib.Path(os.path.join(molecule_names_folder_path, file))
            molecule_names_list = orderly.data.util.load_list(path=file_path)
            if len(molecule_names_list) > 0:
                all_empty = False
    assert not all_empty


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
    toy_dict: Dict[str, List[str]],
    columns_to_count_from: List[str],
    expected_total_value_counts: Dict[str, int],
) -> None:
    import orderly.clean.cleaner
    import pandas as pd
    import copy

    toy_dict = copy.copy(toy_dict)

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
    toy_dict: Dict[str, List[str]],
    columns_to_transform: List[str],
    value_counts_dict: Dict[str, int],
    min_frequency_of_occurrence: int,
    expected_dict: Dict[str, List[str]],
) -> None:
    import pandas as pd
    import orderly.clean.cleaner
    import copy

    toy_dict = copy.copy(toy_dict)

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
    toy_dict: Dict[str, List[str]],
    columns_to_transform: List[str],
    value_counts_dict: Dict[str, int],
    min_frequency_of_occurrence: int,
    expected_dict: Dict[str, List[str]],
) -> None:
    import pandas as pd
    import orderly.clean.cleaner
    import copy

    toy_dict = copy.copy(toy_dict)

    df = pd.DataFrame(toy_dict)
    value_counts_series = pd.Series(value_counts_dict)

    df = orderly.clean.cleaner.Cleaner._remove_rare_molecules(
        df, columns_to_transform, value_counts_series, min_frequency_of_occurrence
    )

    expected_df = pd.DataFrame(expected_dict)

    assert df.equals(expected_df), f"Got: {df}, expected: {expected_df},"


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
def test_get_cleaned_df(cleaned_df_params: Tuple[pd.DataFrame, List[Any]]) -> None:
    import copy

    cleaned_df, _ = copy.copy(cleaned_df_params)
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
def test_number_of_columns(cleaned_df_params: Tuple[pd.DataFrame, List[Any]]) -> None:
    import copy

    cleaned_df, params = copy.copy(cleaned_df_params)

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
    assert 'grant_date' in cols
    assert 'date_of_experiment' in cols
    assert cleaned_df['date_of_experiment'].dtype == 'datetime64' or cleaned_df['date_of_experiment'].dtype is None



def double_list(
    x: List[Any],
) -> Tuple[List[Any], List[Any]]:
    return (x, x)


@pytest.mark.parametrize(
    "cleaned_df_params_without_unresolved_names_and_duplicates,cleaned_df_params_without_min_freq_without_unresolved_names_and_duplicates",
    (
        pytest.param(
            *double_list([False, False, 5, 5, 2, 3, 0, 0, 15, False]),
            id="trust_labelling:F|consistent_yield:F|map_rare_molecules_to_other:F|15",
        ),
        pytest.param(
            *double_list([False, False, 5, 5, 2, 3, 0, 0, 100, False]),
            id="trust_labelling:F|consistent_yield:F|map_rare_molecules_to_other:F|100",
        ),
        pytest.param(
            *double_list([True, False, 5, 5, 2, 0, 2, 1, 15, False]),
            id="trust_labelling:T|consistent_yield:F|map_rare_molecules_to_other:F",
        ),
        pytest.param(
            *double_list([False, True, 5, 5, 2, 3, 0, 0, 15, False]),
            id="trust_labelling:F|consistent_yield:T|map_rare_molecules_to_other:F",
        ),
        pytest.param(
            *double_list([False, False, 5, 5, 2, 3, 0, 0, 15, True]),
            id="trust_labelling:F|consistent_yield:F|map_rare_molecules_to_other:T",
        ),
        pytest.param(
            *double_list([True, True, 5, 5, 2, 0, 2, 1, 15, False]),
            id="trust_labelling:T|consistent_yield:T|map_rare_molecules_to_other:F",
        ),
        pytest.param(
            *double_list([True, False, 5, 5, 2, 0, 2, 1, 15, True]),
            id="trust_labelling:T|consistent_yield:F|map_rare_molecules_to_other:T",
        ),
        pytest.param(
            *double_list([False, True, 5, 5, 2, 3, 0, 0, 15, True]),
            id="trust_labelling:F|consistent_yield:T|map_rare_molecules_to_other:T",
        ),
        pytest.param(
            *double_list([True, True, 5, 5, 2, 0, 2, 1, 15, True]),
            id="trust_labelling:T|consistent_yield:T|map_rare_molecules_to_other:T",
        ),
        # XFAILS
        pytest.param(
            *double_list([False, True, 5, 5, 5, 5, 5, 5, 5, True]),
            marks=pytest.mark.xfail(
                reason="AssertionError: Invalid input: If trust_labelling=True in orderly.extract, then num_cat and num_reag must be 0. If trust_labelling=False, then num_agent must be 0."
            ),
            id="trust_labelling:F|consistent_yield:T|map_rare_molecules_to_other:F|fives",
        ),
    ),
    indirect=True,
)
def test_frequency_without_unresolved_names_and_duplicates(
    cleaned_df_params_without_unresolved_names_and_duplicates: Tuple[
        pd.DataFrame, List[Any]
    ],
    cleaned_df_params_without_min_freq_without_unresolved_names_and_duplicates: Tuple[
        pd.DataFrame, List[Any]
    ],
) -> None:
    import copy

    cleaned_df, params = copy.copy(
        cleaned_df_params_without_unresolved_names_and_duplicates
    )
    uncleaned_df, unclean_params = copy.copy(
        cleaned_df_params_without_min_freq_without_unresolved_names_and_duplicates
    )

    assert len(params) == len(unclean_params)
    assert unclean_params[-2] == 0
    assert len(params) == 10
    min_frequency_of_occurrence = params[-2]

    import pandas as pd

    def get_value_counts(df: pd.DataFrame) -> pd.Series:
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

        total_value_counts = (
            pd.concat(results, axis=0, sort=True).groupby(level=0).sum()
        )
        if "other" in total_value_counts.index:
            total_value_counts = total_value_counts.drop("other")
        total_value_counts = total_value_counts.sort_values(ascending=True)
        return total_value_counts

    cleaned_value_counts = get_value_counts(df=cleaned_df.copy())
    uncleaned_value_counts = get_value_counts(df=uncleaned_df.copy())

    assert min_frequency_of_occurrence > 0  # sanity check the copying worked

    cleaned_rare = cleaned_value_counts[
        cleaned_value_counts < min_frequency_of_occurrence
    ]
    uncleaned_rare = uncleaned_value_counts[
        uncleaned_value_counts < min_frequency_of_occurrence
    ]

    if not cleaned_rare.empty:
        assert uncleaned_rare.index.intersection(cleaned_value_counts.index).empty


@pytest.mark.parametrize(
    "cleaned_df_params,cleaned_df_params_without_min_freq",
    (
        pytest.param(
            *double_list([False, False, 5, 5, 2, 3, 0, 0, 15, False]),
            id="trust_labelling:F|consistent_yield:F|map_rare_molecules_to_other:F|15",
        ),
        pytest.param(
            *double_list([False, False, 5, 5, 2, 3, 0, 0, 100, False]),
            id="trust_labelling:F|consistent_yield:F|map_rare_molecules_to_other:F|100",
        ),
        pytest.param(
            *double_list([True, False, 5, 5, 2, 0, 2, 1, 15, False]),
            id="trust_labelling:T|consistent_yield:F|map_rare_molecules_to_other:F",
        ),
        pytest.param(
            *double_list([False, True, 5, 5, 2, 3, 0, 0, 15, False]),
            id="trust_labelling:F|consistent_yield:T|map_rare_molecules_to_other:F",
        ),
        pytest.param(
            *double_list([True, True, 5, 5, 2, 0, 2, 1, 15, False]),
            id="trust_labelling:T|consistent_yield:T|map_rare_molecules_to_other:F",
        ),
        pytest.param(
            *double_list([False, False, 5, 5, 2, 3, 0, 0, 15, True]),
            id="trust_labelling:F|consistent_yield:F|map_rare_molecules_to_other:T",
        ),
        pytest.param(
            *double_list([True, False, 5, 5, 2, 0, 2, 1, 15, True]),
            id="trust_labelling:T|consistent_yield:F|map_rare_molecules_to_other:T",
        ),
        pytest.param(
            *double_list([False, True, 5, 5, 2, 3, 0, 0, 15, True]),
            id="trust_labelling:F|consistent_yield:T|map_rare_molecules_to_other:T",
        ),
        pytest.param(
            *double_list([True, True, 5, 5, 2, 0, 2, 1, 15, True]),
            id="trust_labelling:T|consistent_yield:T|map_rare_molecules_to_other:T",
        ),
    ),
    indirect=True,
)
def test_frequency_with_unresolved_names_and_duplicates(
    cleaned_df_params: Tuple[pd.DataFrame, List[Any]],
    cleaned_df_params_without_min_freq: Tuple[pd.DataFrame, List[Any]],
) -> None:
    """
    this test checks if enough data is being removed, it is possible too much data is removed but this is covered by other tests
    """
    import copy

    cleaned_df, params = copy.copy(cleaned_df_params)
    uncleaned_df, unclean_params = copy.copy(cleaned_df_params_without_min_freq)

    assert len(params) == len(unclean_params)
    assert unclean_params[-2] == 0
    assert len(params) == 10
    min_frequency_of_occurrence = params[-2]

    import pandas as pd

    def get_value_counts(df: pd.DataFrame) -> pd.Series:
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

        total_value_counts = (
            pd.concat(results, axis=0, sort=True).groupby(level=0).sum()
        )
        if "other" in total_value_counts.index:
            total_value_counts = total_value_counts.drop("other")
        total_value_counts = total_value_counts.sort_values(ascending=True)
        return total_value_counts

    cleaned_value_counts = get_value_counts(df=cleaned_df.copy())
    uncleaned_value_counts = get_value_counts(df=uncleaned_df.copy())

    assert min_frequency_of_occurrence > 0  # sanity check the copying worked

    cleaned_rare = cleaned_value_counts[
        cleaned_value_counts < min_frequency_of_occurrence
    ]
    uncleaned_rare = uncleaned_value_counts[
        uncleaned_value_counts < min_frequency_of_occurrence
    ]

    if not cleaned_rare.empty:
        # if there is stuff that is rare now, we want to make sure it wasnt rare previously
        assert uncleaned_rare.index.intersection(cleaned_value_counts.index).empty
