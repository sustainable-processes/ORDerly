import pytest


def test_hello_world():
    assert True


@pytest.mark.parametrize(
    "rxn_idx,manual_replacements_dict,expected_reactants,expected_reagents,expected_solvents,expected_catalysts,expected_products,expected_yields,expected_temperature,expected_rxn_time,expected_mapped_rxn,expected_names_list,",
    (
        # daniel enter here[rxn here,dict here,expected react here , etc,,,,],
    ),
)
def extract_rxn_extract(
    rxn_idx,
    manual_replacements_dict,
    expected_reactants,
    expected_reagents,
    expected_solvents,
    expected_catalysts,
    expected_products,
    expected_yields,
    expected_temperature,
    expected_rxn_time,
    expected_mapped_rxn,
    expected_names_list,
):
    import orderly.extract.extractor

    # load from rxn_idx
    rxn = rxn_idx  # TODO

    (
        reactants,
        reagents,
        solvents,
        catalysts,
        products,
        yields,
        temperature,
        rxn_time,
        mapped_rxn,
        names_list,
    ) = orderly.extract.extractor.OrdExtractor.handle_reaction_object(
        rxn, manual_replacements_dict
    )
    assert reactants == expected_reactants
    assert reagents == expected_reagents
    assert solvents == expected_solvents
    assert catalysts == expected_catalysts
    assert products == expected_products
    assert yields == expected_yields
    assert temperature == expected_temperature
    assert rxn_time == expected_rxn_time
    assert mapped_rxn == expected_mapped_rxn
    assert names_list == expected_names_list


@pytest.mark.parametrize(
    "merge_conditions,use_multiprocessing,name_contains_substring,inverse_substring",
    (
        [False, True, "uspto", True],
        [False, True, "uspto", False],
        [True, False, "uspto", True],
        [True, True, None, True],
    ),
)
def test_extraction_pipeline(
    tmp_path,
    merge_conditions,
    use_multiprocessing,
    name_contains_substring,
    inverse_substring,
):
    pickled_data_folder = tmp_path / "pkl_data"
    pickled_data_folder.mkdir()
    molecule_names_folder = tmp_path / "molecule_names"
    molecule_names_folder.mkdir()

    import orderly.extract.main
    import orderly.data

    orderly.extract.main.main(
        data_path=str(orderly.data.get_path_of_test_ords()),
        ord_file_ending=".pb.gz",
        merge_conditions=merge_conditions,
        output_path=tmp_path,
        pickled_data_folder=pickled_data_folder,
        molecule_names_folder=molecule_names_folder,
        merged_molecules_file="all_molecule_names.pkl",
        use_multiprocessing=use_multiprocessing,
        name_contains_substring=name_contains_substring,
        inverse_substring=inverse_substring,
        overwrite=False,
    )
