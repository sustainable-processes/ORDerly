import pathlib
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytest


def test_hello_world() -> None:
    assert True


@pytest.fixture
def toy_dict() -> Dict[str, List[str]]:
    toy_dict = {
        "reactant_000": ["B", "A", "F", "A"],
        "reactant_001": ["D", "A", pd.NA, "B"],
        "product_000": ["C", "A", "E", "A"],
        "product_001": ["E", "G", "C", "H"],
        "agent_000": ["D", "F", "D", "B"],
        "agent_001": ["C", "E", "G", "A"],
        "solvent_000": ["E", "B", "G", "C"],
        "solvent_001": ["C", "D", "B", "G"],
        "solvent_002": ["D", "B", "F", "G"],
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
    set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn: bool,
    set_unresolved_names_to_none: bool,
    remove_rxn_with_unresolved_names: bool,
    remove_reactions_with_no_reactants: bool,
    remove_reactions_with_no_products: bool,
    remove_reactions_with_no_solvents: bool,
    remove_reactions_with_no_agents: bool,
    remove_reactions_with_no_conditions: bool,
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
        output_path=output_path,
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
        set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn,
        set_unresolved_names_to_none=set_unresolved_names_to_none,
        remove_rxn_with_unresolved_names=remove_rxn_with_unresolved_names,
        remove_reactions_with_no_reactants=remove_reactions_with_no_reactants,
        remove_reactions_with_no_products=remove_reactions_with_no_products,
        remove_reactions_with_no_solvents=remove_reactions_with_no_solvents,
        remove_reactions_with_no_agents=remove_reactions_with_no_agents,
        remove_reactions_with_no_conditions=remove_reactions_with_no_conditions,
        scramble=False,
        train_size=0,
        drop_duplicates=drop_duplicates,
        disable_tqdm=False,
        overwrite=False,
    )

    import pandas as pd

    return pd.read_parquet(output_path)


@pytest.fixture
def cleaned_df_params_default(
    tmp_path: pathlib.Path, request: pytest.FixtureRequest
) -> Tuple[pd.DataFrame, List[Any]]:
    assert len(request.param) == 10
    updated_args = request.param + [
        True,
        False,
        False,
        True,
        True,
        True,
        True,
        False,
        True,
    ]
    # set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn: bool,
    # set_unresolved_names_to_none: bool,
    # remove_rxn_with_unresolved_names: bool,
    # remove_reactions_with_no_reactants: bool,
    # remove_reactions_with_no_products: bool,
    # remove_reactions_with_no_solvents: bool,
    # remove_reactions_with_no_agents: bool,
    # remove_reactions_with_no_conditions: bool,
    # drop_duplicates: bool
    return (
        get_cleaned_df(
            tmp_path / "cleaned_df" / "orderly_ord.parquet",
            *updated_args,
        ),
        request.param,
    )


@pytest.fixture
def cleaned_df_params_default2(
    tmp_path: pathlib.Path, request: pytest.FixtureRequest
) -> Tuple[pd.DataFrame, List[Any]]:
    """Copy of 'cleaned_df_params_default', but needed a second fixture for one of the tests."""
    assert len(request.param) == 10
    updated_args = request.param + [
        True,
        False,
        False,
        True,
        True,
        True,
        True,
        False,
        True,
    ]
    # set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn: bool,
    # set_unresolved_names_to_none: bool,
    # remove_rxn_with_unresolved_names: bool,
    # remove_reactions_with_no_reactants: bool,
    # remove_reactions_with_no_products: bool,
    # remove_reactions_with_no_solvents: bool,
    # remove_reactions_with_no_agents: bool,
    # remove_reactions_with_no_conditions: bool,
    # drop_duplicates: bool
    return (
        get_cleaned_df(
            tmp_path / "cleaned_df2" / "orderly_ord.parquet",
            *updated_args,
        ),
        request.param,
    )


@pytest.fixture
def cleaned_df_params_default_without_min_freq(
    tmp_path: pathlib.Path, request: pytest.FixtureRequest
) -> Tuple[pd.DataFrame, List[Any]]:
    import copy

    args = copy.copy(request.param)
    args[-2] = 0
    assert len(request.param) == 10
    updated_args = args + [
        True,
        False,
        False,
        True,
        True,
        True,
        True,
        False,
        True,
    ]
    # set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn: bool,
    # set_unresolved_names_to_none: bool,
    # remove_rxn_with_unresolved_names: bool,
    # remove_reactions_with_no_reactants: bool,
    # remove_reactions_with_no_products: bool,
    # remove_reactions_with_no_solvents: bool,
    # remove_reactions_with_no_agents: bool,
    # remove_reactions_with_no_conditions: bool,
    # drop_duplicates: bool
    return (
        get_cleaned_df(
            tmp_path
            / "cleaned_df_params_default_without_min_freq"
            / "orderly_ord.parquet",
            *updated_args,
        ),
        args,
    )


@pytest.fixture
def cleaned_df_params_retaining_unresolved_names_and_duplicates(
    tmp_path: pathlib.Path, request: pytest.FixtureRequest
) -> Tuple[pd.DataFrame, List[Any]]:
    assert len(request.param) == 10
    updated_args = request.param + [
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    # set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn: bool,
    # set_unresolved_names_to_none: bool,
    # remove_rxn_with_unresolved_names: bool,
    # remove_reactions_with_no_reactants: bool,
    # remove_reactions_with_no_products: bool,
    # remove_reactions_with_no_solvents: bool,
    # remove_reactions_with_no_agents: bool,
    # remove_reactions_with_no_conditions: bool,
    # drop_duplicates: bool

    return (
        get_cleaned_df(
            tmp_path
            / "cleaned_df_params_retaining_unresolved_names_and_duplicates"
            / "orderly_ord.parquet",
            *updated_args,
        ),
        request.param,
    )


@pytest.fixture
def cleaned_df_params_retaining_unresolved_names_and_duplicates_without_min_freq(
    tmp_path: pathlib.Path, request: pytest.FixtureRequest
) -> Tuple[pd.DataFrame, List[Any]]:
    import copy

    args = copy.copy(request.param)
    args[-2] = 0
    assert len(request.param) == 10
    updated_args = args + [
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    # set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn: bool,
    # set_unresolved_names_to_none: bool,
    # remove_rxn_with_unresolved_names: bool,
    # remove_reactions_with_no_reactants: bool,
    # remove_reactions_with_no_products: bool,
    # remove_reactions_with_no_solvents: bool,
    # remove_reactions_with_no_agents: bool,
    # remove_reactions_with_no_conditions: bool,
    # drop_duplicates: bool
    return (
        get_cleaned_df(
            tmp_path
            / "cleaned_df_params_retaining_unresolved_names_and_duplicates_without_min_freq"
            / "orderly_ord.parquet",
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
                "reactant_000",
                "reactant_001",
                "product_000",
                "product_001",
                "agent_000",
                "agent_001",
                "solvent_000",
                "solvent_001",
                "solvent_002",
            ],
            {"A": 6, "B": 6, "C": 5, "D": 5, "E": 4, "F": 3, "G": 5, "H": 1},
            id="all_columns",
        ),
        pytest.param(
            ["agent_000", "agent_001", "solvent_000", "solvent_001", "solvent_002"],
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
    import copy

    import pandas as pd

    import orderly.clean.cleaner

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


def test_scramble(
    toy_dict: Dict[str, List[str]],
) -> None:
    import copy

    import pandas as pd

    import orderly.clean.cleaner

    toy_dict = copy.copy(toy_dict)

    df = pd.DataFrame(toy_dict)
    components = ("reactant", "agent", "solvent", "catalyst", "reagent")

    scrambled_df = orderly.clean.cleaner.Cleaner._scramble(df, components)

    assert not scrambled_df.equals(
        df
    ), f"Got: {scrambled_df}, expected to be different from: {df},"

    # check that molecules have only been moved around within the same reaction (i.e. the same row)
    rows_to_check = min(10, len(df))
    for i in range(rows_to_check):
        sorted_row_components = set(df.iloc[i])
        scrambled_row_components = set(scrambled_df.iloc[i])
        assert (
            sorted_row_components == scrambled_row_components
        ), f"Got: {sorted_row_components}, expected: {scrambled_row_components},"


@pytest.mark.parametrize(
    "component_name, number_of_columns_to_keep, expected_dict",
    (
        # Del reactions with too many reactants
        pytest.param(
            "reactant",
            1,
            {
                "reactant_000": ["F"],
                "product_000": ["E"],
                "product_001": ["C"],
                "agent_000": ["D"],
                "agent_001": ["G"],
                "solvent_000": ["G"],
                "solvent_001": ["B"],
                "solvent_002": ["F"],
            },
        ),
        # Add a component
        pytest.param(
            "reactant",
            3,
            {
                "reactant_000": ["B", "A", "F", "A"],
                "reactant_001": ["D", "A", pd.NA, "B"],
                "reactant_002": [pd.NA, pd.NA, pd.NA, pd.NA],
                "product_000": ["C", "A", "E", "A"],
                "product_001": ["E", "G", "C", "H"],
                "agent_000": ["D", "F", "D", "B"],
                "agent_001": ["C", "E", "G", "A"],
                "solvent_000": ["E", "B", "G", "C"],
                "solvent_001": ["C", "D", "B", "G"],
                "solvent_002": ["D", "B", "F", "G"],
            },
        ),
        # Add two component
        pytest.param(
            "reactant",
            4,
            {
                "reactant_000": ["B", "A", "F", "A"],
                "reactant_001": ["D", "A", pd.NA, "B"],
                "reactant_002": [pd.NA, pd.NA, pd.NA, pd.NA],
                "reactant_003": [pd.NA, pd.NA, pd.NA, pd.NA],
                "product_000": ["C", "A", "E", "A"],
                "product_001": ["E", "G", "C", "H"],
                "agent_000": ["D", "F", "D", "B"],
                "agent_001": ["C", "E", "G", "A"],
                "solvent_000": ["E", "B", "G", "C"],
                "solvent_001": ["C", "D", "B", "G"],
                "solvent_002": ["D", "B", "F", "G"],
            },
        ),
        # Do nothing
        pytest.param(
            "reactant",
            2,
            {
                "reactant_000": ["B", "A", "F", "A"],
                "reactant_001": ["D", "A", pd.NA, "B"],
                "product_000": ["C", "A", "E", "A"],
                "product_001": ["E", "G", "C", "H"],
                "agent_000": ["D", "F", "D", "B"],
                "agent_001": ["C", "E", "G", "A"],
                "solvent_000": ["E", "B", "G", "C"],
                "solvent_001": ["C", "D", "B", "G"],
                "solvent_002": ["D", "B", "F", "G"],
            },
        ),
        # Do nothing
        pytest.param(
            "reactant",
            -1,
            {
                "reactant_000": ["B", "A", "F", "A"],
                "reactant_001": ["D", "A", pd.NA, "B"],
                "product_000": ["C", "A", "E", "A"],
                "product_001": ["E", "G", "C", "H"],
                "agent_000": ["D", "F", "D", "B"],
                "agent_001": ["C", "E", "G", "A"],
                "solvent_000": ["E", "B", "G", "C"],
                "solvent_001": ["C", "D", "B", "G"],
                "solvent_002": ["D", "B", "F", "G"],
            },
        ),
    ),
)
def test_remove_reactions_with_too_many_of_component(
    toy_dict: Dict[str, List[str]],
    component_name: str,
    number_of_columns_to_keep: int,
    expected_dict: Dict[str, int],
) -> None:
    import copy

    import pandas as pd

    import orderly.clean.cleaner

    toy_dict = copy.copy(toy_dict)

    df = pd.DataFrame(toy_dict)

    filtered_df = (
        orderly.clean.cleaner.Cleaner._remove_reactions_with_too_many_of_component(
            df=df,
            component_name=component_name,
            number_of_columns_to_keep=number_of_columns_to_keep,
        )
    )

    expected_filtered_df = pd.DataFrame(expected_dict).sort_index(axis=1)
    if filtered_df.empty:
        assert filtered_df.empty == expected_filtered_df.empty
    else:
        assert filtered_df.equals(
            expected_filtered_df
        ), f"Got: {filtered_df}, expected: {expected_filtered_df},"


@pytest.mark.parametrize(
    "columns_to_transform, value_counts_dict, min_frequency_of_occurrence, expected_dict,",
    (
        pytest.param(
            [
                "reactant_000",
                "reactant_001",
                "product_000",
                "product_001",
                "agent_000",
                "agent_001",
                "solvent_000",
                "solvent_001",
                "solvent_002",
            ],
            {"A": 6, "B": 6, "C": 5, "D": 5, "E": 4, "F": 3, "G": 5, "H": 1},
            4,
            {
                "reactant_000": ["B", "A", "other", "A"],
                "reactant_001": ["D", "A", pd.NA, "B"],
                "product_000": ["C", "A", "E", "A"],
                "product_001": ["E", "G", "C", "other"],
                "agent_000": ["D", "other", "D", "B"],
                "agent_001": ["C", "E", "G", "A"],
                "solvent_000": ["E", "B", "G", "C"],
                "solvent_001": ["C", "D", "B", "G"],
                "solvent_002": ["D", "B", "other", "G"],
            },
            id="all_columns",
        ),
        pytest.param(
            ["agent_000", "agent_001", "solvent_000", "solvent_001", "solvent_002"],
            {"A": 1, "B": 4, "C": 3, "D": 4, "E": 2, "F": 2, "G": 3},
            3,
            {
                "reactant_000": ["B", "A", "F", "A"],
                "reactant_001": ["D", "A", pd.NA, "B"],
                "product_000": ["C", "A", "E", "A"],
                "product_001": ["E", "G", "C", "H"],
                "agent_000": ["D", "other", "D", "B"],
                "agent_001": ["C", "other", "G", "other"],
                "solvent_000": ["other", "B", "G", "C"],
                "solvent_001": ["C", "D", "B", "G"],
                "solvent_002": ["D", "B", "other", "G"],
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
    import copy

    import pandas as pd

    import orderly.clean.cleaner

    toy_dict = copy.copy(toy_dict)

    df = pd.DataFrame(toy_dict)
    value_counts_series = pd.Series(value_counts_dict)

    df = orderly.clean.cleaner.Cleaner._map_rare_molecules_to_other(
        df, columns_to_transform, value_counts_series, min_frequency_of_occurrence
    )

    expected_df = pd.DataFrame(expected_dict)

    assert df.equals(expected_df), f"Got: {df}, expected: {expected_df},"


@pytest.mark.parametrize(
    "toy_dict, target_strings, expected_dict,",
    (
        pytest.param(
            {
                "reactant_000": [
                    "a",
                    None,
                    None,
                    "a",
                ],
                "reactant_001": [
                    None,
                    "a",
                    None,
                    "b",
                ],
                "reactant_002": [
                    "b",
                    "b",
                    "a",
                    "c",
                ],
                "agent_000": [
                    "a",
                    "a",
                    None,
                    None,
                ],
                "agent_001": [
                    None,
                    "b",
                    "a",
                    None,
                ],
                "agent_002": [
                    "b",
                    "c",
                    "b",
                    None,
                ],
            },
            ("reactant", "product", "agent", "solvent"),
            {
                "reactant_000": [
                    "a",
                    "a",
                    "a",
                    "a",
                ],
                "reactant_001": [
                    "b",
                    "b",
                    None,
                    "b",
                ],
                "reactant_002": [
                    None,
                    None,
                    None,
                    "c",
                ],
                "agent_000": [
                    "a",
                    "a",
                    "a",
                    None,
                ],
                "agent_001": [
                    "b",
                    "b",
                    "b",
                    None,
                ],
                "agent_002": [
                    None,
                    "c",
                    None,
                    None,
                ],
            },
        ),
        pytest.param(
            {
                "product_000": [
                    "a",
                    "a",
                    None,
                    None,
                ],
                "product_001": [
                    None,
                    "b",
                    "a",
                    None,
                ],
                "product_002": [
                    "b",
                    "c",
                    "b",
                    None,
                ],
                "yield_000": [
                    "a",
                    "a",
                    "c",
                    None,
                ],
                "yield_001": [
                    None,
                    "b",
                    "a",
                    None,
                ],
                "yield_002": [
                    "b",
                    "c",
                    "b",
                    None,
                ],
            },
            ("product",),
            {
                "product_000": [
                    "a",
                    "a",
                    "a",
                    None,
                ],
                "product_001": [
                    "b",
                    "b",
                    "b",
                    None,
                ],
                "product_002": [
                    None,
                    "c",
                    None,
                    None,
                ],
                "yield_000": [
                    "a",
                    "a",
                    "a",
                    None,
                ],
                "yield_001": [
                    "b",
                    "b",
                    "b",
                    None,
                ],
                "yield_002": [
                    None,
                    "c",
                    "c",
                    None,
                ],
            },
        ),
    ),
)
def test_move_none_to_after_data(
    toy_dict: Dict[str, List[str]],
    target_strings: Tuple[str, ...],
    expected_dict: Dict[str, List[str]],
) -> None:
    import copy

    import pandas as pd

    import orderly.clean.cleaner

    toy_dict = copy.copy(toy_dict)

    df = pd.DataFrame(toy_dict)

    df = orderly.clean.cleaner.Cleaner._move_none_to_after_data(df, target_strings)

    expected_df = pd.DataFrame(expected_dict)

    assert df.equals(expected_df), f"Got: \n{df}, expected: \n{expected_df},"


@pytest.mark.parametrize(
    "columns_to_transform, value_counts_dict, min_frequency_of_occurrence, expected_dict,",
    (
        pytest.param(
            [
                "reactant_000",
                "reactant_001",
                "product_000",
                "product_001",
                "agent_000",
                "agent_001",
                "solvent_000",
                "solvent_001",
                "solvent_002",
            ],
            {"A": 6, "B": 6, "C": 5, "D": 5, "E": 4, "F": 3, "G": 6, "H": 1},
            4,
            {
                "reactant_000": ["B"],
                "reactant_001": ["D"],
                "product_000": ["C"],
                "product_001": ["E"],
                "agent_000": ["D"],
                "agent_001": ["C"],
                "solvent_000": ["E"],
                "solvent_001": ["C"],
                "solvent_002": ["D"],
            },
            id="all_columns",
        ),
        pytest.param(
            ["agent_000", "agent_001", "solvent_000", "solvent_001", "solvent_002"],
            {"A": 1, "B": 4, "C": 3, "D": 4, "E": 2, "F": 2, "G": 4},
            2,
            {
                "reactant_000": ["B", "A", "F"],
                "reactant_001": ["D", "A", pd.NA],
                "product_000": ["C", "A", "E"],
                "product_001": ["E", "G", "C"],
                "agent_000": ["D", "F", "D"],
                "agent_001": ["C", "E", "G"],
                "solvent_000": ["E", "B", "G"],
                "solvent_001": ["C", "D", "B"],
                "solvent_002": ["D", "B", "F"],
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
    import copy

    import pandas as pd

    import orderly.clean.cleaner

    toy_dict = copy.copy(toy_dict)

    df = pd.DataFrame(toy_dict)
    value_counts_series = pd.Series(value_counts_dict)

    df = orderly.clean.cleaner.Cleaner._remove_rare_molecules(
        df, columns_to_transform, value_counts_series, min_frequency_of_occurrence
    )

    expected_df = pd.DataFrame(expected_dict)

    assert df.equals(expected_df), f"Got: {df}, expected: {expected_df},"


@pytest.mark.parametrize(
    "cleaned_df_params_default",
    (
        pytest.param(
            [False, False, 5, 5, 2, 3, 0, 0, 55, False],
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
def test_get_cleaned_df(
    cleaned_df_params_default: Tuple[pd.DataFrame, List[Any]]
) -> None:
    import copy

    cleaned_df, _ = copy.copy(cleaned_df_params_default)
    assert not cleaned_df.empty
    # TODO: check that there's only NaN or NaT, but no None


@pytest.mark.parametrize(
    "cleaned_df_params_default",
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
def test_number_of_columns_and_order_of_None(
    cleaned_df_params_default: Tuple[pd.DataFrame, List[Any]]
) -> None:
    import copy

    import numpy as np

    cleaned_df, params = copy.copy(cleaned_df_params_default)

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

    # if num_reactant == -1 we just include all reactant columns (and same for the others)
    assert (num_reactant_cols == num_reactant) or num_reactant == -1
    assert (num_product_cols == num_product) or num_product == -1
    assert (num_agent_cols == num_agent) or num_agent == -1
    assert (num_cat_cols == num_cat) or num_cat == -1
    assert (num_reag_cols == num_reag) or num_reag == -1
    assert (num_solv_cols == num_solv) or num_solv == -1

    assert "grant_date" in cols
    assert "date_of_experiment" in cols

    import numpy as np

    grant_date = cleaned_df["grant_date"].replace({None: np.nan})
    date_of_experiment = cleaned_df["date_of_experiment"].replace({None: np.nan})
    if not grant_date.dropna().empty:
        assert pd.api.types.is_datetime64_ns_dtype(
            grant_date
        ), f"failure for grant_date: {cleaned_df['grant_date'].dtype}"
    if not date_of_experiment.dropna().empty:
        assert pd.api.types.is_datetime64_ns_dtype(
            date_of_experiment
        ), f"failure for date_of_experiment: {cleaned_df['date_of_experiment'].dtype}"

    # Check that grant_date and date_of_experiment columns are either dtype = datetime64 or full of None
    datetime_coumns = cleaned_df.select_dtypes(include=[np.datetime64])
    assert ("grant_date" in datetime_coumns.columns) or (
        len(cleaned_df["grant_date"].dropna()) == 0
    )
    assert ("date_of_experiment" in datetime_coumns.columns) or (
        len(cleaned_df["date_of_experiment"].dropna()) == 0
    )

    # Also check that there are no instances of None before data in the cleaned df

    def _get_columns_beginning_with_str(
        columns: List[str], target_strings: Optional[Tuple[str, ...]] = None
    ) -> List[str]:
        """goes through the column in a dataframe and adds columns that start with a string in the target strings"""
        if target_strings is None:
            target_strings = (
                "agent",
                "solvent",
                "reagent",
                "catalyst",
                "product",
                "reactant",
            )

        return sorted([col for col in columns if col.startswith(target_strings)])

    target_strings = (
        "agent",
        "solvent",
        "reagent",
        "catalyst",
        "product",
        "reactant",
    )

    def check_valid_order(row: pd.Series) -> pd.Series:
        seen_none = False
        for idx, a in enumerate(row):
            current_isna = pd.isna(a) or (a == "")
            if seen_none:
                if not current_isna:
                    raise ValueError(f"Unexpected order at {idx=} for {row.tolist()=}")
            if current_isna:
                seen_none = True
        return row

    for target_string in target_strings:
        target_columns = _get_columns_beginning_with_str(
            columns=cleaned_df.columns,
            target_strings=(target_string,),
        )
        # check that there are no instances of None before data in the cleaned df
        for idx, row in cleaned_df.loc[:, target_columns].iterrows():
            check_valid_order(row)
        # cleaned_df.loc[:, target_columns].apply(check_valid_order, axis=1)


def double_list(
    x: List[Any],
) -> Tuple[List[Any], List[Any]]:
    return (x, x)


@pytest.mark.parametrize(
    "cleaned_df_params_default,cleaned_df_params_default_without_min_freq",
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
def test_frequency_params_default(
    cleaned_df_params_default: Tuple[pd.DataFrame, List[Any]],
    cleaned_df_params_default_without_min_freq: Tuple[pd.DataFrame, List[Any]],
) -> None:
    """
    Test that there are no rare molecules left in the dataset after the cleaning process. There may be molecules appearing less often than the minimum frequency of occurrence, even after the cleaning, but this is because when deleting rows with rare values we may cause new molecules to become rare; this is why we check for the intersection of the two dfs at the very end of the test.
    """

    import copy

    cleaned_df, params = copy.copy(cleaned_df_params_default)
    uncleaned_df, unclean_params = copy.copy(cleaned_df_params_default_without_min_freq)

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
    assert "" not in cleaned_value_counts.keys()  # check there are no empty strings
    assert "" not in uncleaned_value_counts.keys()  # check there are no empty strings

    cleaned_rare = cleaned_value_counts[
        cleaned_value_counts < min_frequency_of_occurrence
    ]
    uncleaned_rare = uncleaned_value_counts[
        uncleaned_value_counts < min_frequency_of_occurrence
    ]

    if not cleaned_rare.empty:
        assert uncleaned_rare.index.intersection(cleaned_value_counts.index).empty

    consistent_yield = params[1]
    if consistent_yield:
        # Check there's no instance of na and None in the yield column
        num_products = params[3]
        
        # Assuming num_products is defined and cleaned_df is your DataFrame
        yield_columns = [f"yield_{i:03d}" for i in range(num_products)]

        # Summing over the specified columns for each row and creating a new column 'total yield'
        cleaned_df['total yield'] = cleaned_df[yield_columns].sum(axis=1)
        
        assert all(cleaned_df['total yield'].apply(lambda x: 0 <= x <= 100)), "Total yield values are not all floats in range 0-100"
        


#####


@pytest.mark.parametrize(
    "cleaned_df_params_default,cleaned_df_params_default2",
    (
        pytest.param(
            [True, False, 5, 5, 2, 0, 2, 1, 15, False],
            [False, False, 5, 5, 2, 3, 0, 0, 15, False],
            id="trust_labelling:T/F|consistent_yield:F|map_rare_molecules_to_other:F|15",
        ),
        pytest.param(
            [False, False, 5, 5, 2, 3, 0, 0, 5, False],
            [False, False, 5, 5, 2, 3, 0, 0, 15, False],
            id="trust_labelling:F|consistent_yield:F|map_rare_molecules_to_other:F|5/15",
        ),
    ),
    indirect=["cleaned_df_params_default", "cleaned_df_params_default2"],
)
def test_original_index(
    cleaned_df_params_default: Tuple[pd.DataFrame, List[Any]],
    cleaned_df_params_default2: Tuple[pd.DataFrame, List[Any]],
) -> None:
    """
    Test that the reaction referred to with "original_index" is the same accross different ways of cleaning the dataset.
    """

    import copy

    df1, params1 = copy.copy(cleaned_df_params_default)
    df2, params2 = copy.copy(cleaned_df_params_default2)

    assert len(params1) == len(params2)
    assert len(params1) == 10

    import pandas as pd

    # find an "original_index" they all have in common
    indices_in_common = set(df1["original_index"]).intersection(
        set(df2["original_index"])
    )
    for i in indices_in_common:
        a = df1[df1["original_index"] == i]["rxn_str"].iloc[0]
        b = df2[df2["original_index"] == i]["rxn_str"].iloc[0]
        assert a == b


#####


@pytest.mark.parametrize(
    "cleaned_df_params_retaining_unresolved_names_and_duplicates,cleaned_df_params_retaining_unresolved_names_and_duplicates_without_min_freq",
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
def test_frequency_retaining_unresolved_names_and_duplicates(
    cleaned_df_params_retaining_unresolved_names_and_duplicates: Tuple[
        pd.DataFrame, List[Any]
    ],
    cleaned_df_params_retaining_unresolved_names_and_duplicates_without_min_freq: Tuple[
        pd.DataFrame, List[Any]
    ],
) -> None:
    """
    Test that there are no rare molecules left in the dataset after the cleaning process. There may be molecules appearing less often than the minimum frequency of occurrence, even after the cleaning, but this is because when deleting rows with rare values we may cause new molecules to become rare; this is why we check for the intersection of the two dfs at the very end of the test.

    In this test we retain duplicate reactions and unresolved names (e.g. english names for molecules that aren't in our manual mapping).
    """
    import copy

    cleaned_df, params = copy.copy(
        cleaned_df_params_retaining_unresolved_names_and_duplicates
    )
    uncleaned_df, unclean_params = copy.copy(
        cleaned_df_params_retaining_unresolved_names_and_duplicates_without_min_freq
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
    assert "" not in cleaned_value_counts.keys()  # check there are no empty strings
    assert "" not in uncleaned_value_counts.keys()  # check there are no empty strings

    cleaned_rare = cleaned_value_counts[
        cleaned_value_counts < min_frequency_of_occurrence
    ]
    uncleaned_rare = uncleaned_value_counts[
        uncleaned_value_counts < min_frequency_of_occurrence
    ]

    if not cleaned_rare.empty:
        # if there is stuff that is rare now, we want to make sure it wasnt rare previously
        assert uncleaned_rare.index.intersection(cleaned_value_counts.index).empty


def test_move_rows_from_test_to_train_set() -> None:
    """
    Test that the function that moves rows from the test set to the train set works as expected.
    """
    import numpy as np
    import pandas as pd

    import orderly.clean.cleaner

    reactant_columns = ["reactant_000", "reactant_001"]
    product_columns = ["product_000"]

    train_indices = np.array([0, 1, 2])
    test_indices = np.array([3, 4, 5])

    train_dict = {
        "reactant_000": ["a", "b", "c"],
        "reactant_001": ["b", "e", "f"],
        "product_000": ["c", "h", "i"],
    }
    test_dict = {
        "reactant_000": ["a", "b", "c"],
        "reactant_001": ["b", "a", "a"],
        "product_000": ["c", "c", "b"],
    }
    train_df = pd.DataFrame(train_dict)
    test_df = pd.DataFrame(test_dict)
    df = pd.concat([train_df, test_df], axis=0, sort=False)

    df = df.reset_index(drop=True)

    matching_indices = orderly.clean.cleaner.get_matching_indices(
        df, train_indices, test_indices, reactant_columns, product_columns
    )

    assert np.equal(matching_indices, np.array([3, 4])).all()
