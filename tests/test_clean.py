import pytest


def test_hello_world():
    assert True


def test_clean(
    tmp_path
):

    import orderly.clean.cleaner
    import orderly.data
    
    orderly.clean.cleaner.main(
        clean_data_path=tmp_path / "cleaned_USPTO.parquet",
        pickles_path=orderly.data.get_path_of_test_extracted_ords() / "pickled_data",
        molecules_to_remove_path=orderly.data.get_path_of_test_extracted_ords() / "all_molecule_names.pkl",
        consistent_yield=True,
        num_reactant=5,
        num_product=5,
        num_solv=2,
        num_agent=3,
        num_cat=0,
        num_reag=0,
        min_frequency_of_occurance_primary=15,
        min_frequency_of_occurance_secondary=15,
        include_other_category=True,
        map_rare_to_other=3,
        disable_tqdm=False,
    )
