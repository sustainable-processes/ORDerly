import pytest


def test_hello_world():
    assert True


def get_cleaned_df(
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

    pickles_path=orderly.data.get_path_of_test_extracted_ords() / "pickled_data"
    molecules_to_remove_path=orderly.data.get_path_of_test_extracted_ords() / "all_molecule_names.pkl"

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
def cleaned_df(tmp_path, request):
    return get_cleaned_df(tmp_path, *request.param)


@pytest.mark.parametrize(
    'cleaned_df',
    (
        pytest.param([True, 5, 5, 2, 3, 0, 0, 15, 15, True, 3], id="consistent_yield:T|include_other_category:T"),
        pytest.param([False, 5, 5, 2, 3, 0, 0, 15, 15, True, 3], id="consistent_yield:F|include_other_category:T"),
        pytest.param([True, 5, 5, 2, 3, 0, 0, 15, 15, False, 3], id="consistent_yield:T|include_other_category:F"),
        pytest.param([False, 5, 5, 2, 3, 0, 0, 15, 15, False, 3], id="consistent_yield:F|include_other_category:F"),
    ),
    indirect=True
)
def test_get_cleaned_df(cleaned_df):
    assert not cleaned_df.empty

@pytest.mark.parametrize(
    'cleaned_df',
    (
        pytest.param([True, 5, 5, 2, 3, 0, 0, 15, 15, True, 3], id="consistent_yield:T|include_other_category:T"),
        pytest.param([False, 5, 5, 2, 3, 0, 0, 15, 15, True, 3], id="consistent_yield:F|include_other_category:T"),
        pytest.param([True, 5, 5, 2, 3, 0, 0, 15, 15, False, 3], id="consistent_yield:T|include_other_category:F"),
        pytest.param([False, 5, 5, 2, 3, 0, 0, 15, 15, False, 3], id="consistent_yield:F|include_other_category:F"),
    ),
    indirect=True
)
def test_some_length(cleaned_df):
    assert len(cleaned_df) > 0

