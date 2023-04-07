import pytest


def test_hello_world():
    assert True


# TODO: add test checking that min_frequency_of_occurance_primary and min_frequency_of_occurance_secondary are respected
@pytest.mark.parametrize(
    "consistent_yield,num_reactant,num_product,num_solv,num_agent,num_cat,num_reag,min_frequency_of_occurance_primary,min_frequency_of_occurance_secondary,include_other_category,map_rare_to_other_threshold",
    (
        [True, 5, 5, 2, 3, 0, 0, 15, 15, True, 3],
        [False, 5, 5, 2, 3, 0, 0, 15, 15, True, 3],
        [True, 5, 5, 2, 3, 0, 0, 15, 15, False, 3],
        [False, 5, 5, 2, 3, 0, 0, 15, 15, False, 3],
    ),
)
def test_clean(
    tmp_path,
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

    orderly.clean.cleaner.main(
        clean_data_path=tmp_path / "orderly_ord.parquet",
        pickles_path=orderly.data.get_path_of_test_extracted_ords() / "pickled_data",
        molecules_to_remove_path=orderly.data.get_path_of_test_extracted_ords()
        / "all_molecule_names.pkl",
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
