import typing
import pytest


def test_hello_world():
    assert True


def get_rxn_func() -> typing.Callable:
    from ord_schema.proto import reaction_pb2 as ord_reaction_pb2

    def get_rxn(
        file_idx: int,
        rxn_idx: int,
    ) -> ord_reaction_pb2.Reaction:
        import orderly.extract.extractor
        import orderly.data
        import orderly.extract.main

        files = orderly.extract.main.get_file_names(
            directory=orderly.data.get_path_of_test_ords(), file_ending=".pb.gz"
        )

        dataset = orderly.extract.extractor.OrdExtractor.load_data(files[file_idx])
        rxn = dataset.reactions[rxn_idx]
        return rxn

    return get_rxn


@pytest.mark.parametrize(
    "file_idx,rxn_idx,expected_labelled_reactants,expected_labelled_reagents,expected_labelled_solvents,expected_labelled_catalysts,expected_labelled_products_from_input,expected_non_smiles_names_list_additions",
    (
        # [0,0, expected_labelled_reactants,expected_labelled_reagents,expected_labelled_solvents,expected_labelled_catalysts,expected_labelled_products_from_input,expected_non_smiles_names_list_additions],
        # [0,1, expected_labelled_reactants,expected_labelled_reagents,expected_labelled_solvents,expected_labelled_catalysts,expected_labelled_products_from_input,expected_non_smiles_names_list_additions],
    ),
)
def rxn_input_extractor(
    # def test_rxn_input_extractor(
    file_idx,
    rxn_idx,
    expected_labelled_reactants,
    expected_labelled_reagents,
    expected_labelled_solvents,
    expected_labelled_catalysts,
    expected_labelled_products_from_input,
    expected_non_smiles_names_list_additions,
):
    rxn = get_rxn_func()(file_idx=file_idx, rxn_idx=rxn_idx)

    import orderly.extract.extractor

    (
        labelled_reactants,
        labelled_reagents,
        labelled_solvents,
        labelled_catalysts,
        labelled_products_from_input,  # Daniel: I'm not sure what to do with this, it doesn't make sense for people to have put a product as an input, so this list should be empty anyway
        non_smiles_names_list_additions,
    ) = orderly.extract.extractor.OrdExtractor.rxn_input_extractor(rxn)

    assert expected_labelled_reactants == labelled_reactants
    assert expected_labelled_reagents == labelled_reagents
    assert expected_labelled_solvents == labelled_solvents
    assert expected_labelled_catalysts == labelled_catalysts
    assert expected_labelled_products_from_input == labelled_products_from_input
    assert expected_non_smiles_names_list_additions == non_smiles_names_list_additions


@pytest.mark.parametrize(
    "file_idx,rxn_idx,expected_yields,expected_labelled_products,expected_non_smiles_names_list_additions",
    (
        # [0,0, expected_yields,expected_labelled_products,expected_non_smiles_names_list_additions],
        # [0,1, expected_yields,expected_labelled_products,expected_non_smiles_names_list_additions],
    ),
)
def rxn_outcomes_extractor(
    file_idx,
    rxn_idx,
    expected_yields,
    expected_labelled_products,
    expected_non_smiles_names_list_additions,
):
    rxn = get_rxn_func()(file_idx=file_idx, rxn_idx=rxn_idx)

    import orderly.extract.extractor

    (
        yields,
        labelled_products,
        non_smiles_names_list_additions,
    ) = orderly.extract.extractor.OrdExtractor.rxn_outcomes_extractor(rxn)

    assert expected_yields == yields
    assert expected_labelled_products == labelled_products
    assert expected_non_smiles_names_list_additions == non_smiles_names_list_additions


@pytest.mark.parametrize(
    "file_idx,rxn_idx,expected_rxn_str,expected_is_mapped",
    (
        # [0,0, expected_rxn_str,expected_is_mapped],
        # [0,1, expected_rxn_str,expected_is_mapped],
    ),
)
def rxn_string_and_is_mapped(
    file_idx,
    rxn_idx,
    expected_rxn_str,
    expected_is_mapped,
):
    rxn = get_rxn_func()(file_idx=file_idx, rxn_idx=rxn_idx)

    import orderly.extract.extractor

    (
        rxn_str,
        is_mapped,
    ) = orderly.extract.extractor.OrdExtractor.get_rxn_string_and_is_mapped(rxn)

    assert expected_rxn_str == rxn_str
    assert expected_is_mapped == is_mapped


@pytest.mark.parametrize(
    "file_idx,rxn_idx,expected_rxn_info",
    (
        # [0,0, expected_rxn_info],
        # [0,1, expected_rxn_info],
    ),
)
def extract_info_from_rxn(
    file_idx,
    rxn_idx,
    expected_rxn_info,
):
    rxn = get_rxn_func()(file_idx=file_idx, rxn_idx=rxn_idx)

    import orderly.extract.extractor

    rxn_info = orderly.extract.extractor.OrdExtractor.extract_info_from_rxn(rxn)

    assert expected_rxn_info == rxn_info


@pytest.mark.parametrize(
    "file_idx,rxn_idx,expected_temperature",
    (
        # [0,0, expected_temperature],
        # [0,1, expected_temperature],
    ),
)
def temperature_extractor(file_idx, rxn_idx, expected_temperature):
    rxn = get_rxn_func()(file_idx=file_idx, rxn_idx=rxn_idx)

    import orderly.extract.extractor

    temperature = orderly.extract.extractor.OrdExtractor.temperature_extractor(rxn)

    assert expected_temperature == temperature


@pytest.mark.parametrize(
    "file_idx,rxn_idx,expected_rxn_time",
    (
        # [0,0, expected_rxn_time],
        # [0,1, expected_rxn_time],
    ),
)
def time_extractor(
    file_idx,
    rxn_idx,
    expected_rxn_time,
):
    rxn = get_rxn_func()(file_idx=file_idx, rxn_idx=rxn_idx)

    import orderly.extract.extractor

    rxn_time = orderly.extract.extractor.OrdExtractor.rxn_time_extractor(rxn)

    assert expected_rxn_time == rxn_time


@pytest.mark.parametrize(
    "file_idx,rxn_idx,expected_rxn_time",
    (
        # [0, 0, expected_rxn_time],
        # [0, 1, expected_rxn_time],
    ),
)
def merge_to_agents(
    rxn_str_agents,
    labelled_catalysts,
    labelled_solvents,
    labelled_reagents,
    metals,
    solvents_set,
    expected_agents,
    expected_solvents,
):
    import orderly.extract.extractor

    agents, solvents = orderly.extract.extractor.OrdExtractor.merge_to_agents(
        rxn_str_agents,
        labelled_catalysts,
        labelled_solvents,
        labelled_reagents,
        metals,
        solvents_set,
    )

    assert expected_agents == agents
    assert expected_solvents == solvents


@pytest.mark.parametrize(
    "rxn_str_products,labelled_products,input_yields,expected_products,expected_yields",
    (
        # [rxn_str_products,labelled_products,input_yields,expected_products,expected_yields],
    ),
)
def match_yield_with_product(
    rxn_str_products,
    labelled_products,
    input_yields,
    expected_products,
    expected_yields,
):
    import orderly.extract.extractor

    products, yields = orderly.extract.extractor.OrdExtractor.match_yield_with_product(
        rxn_str_products, labelled_products, input_yields
    )

    assert expected_products == products
    assert expected_yields == yields


@pytest.mark.parametrize(
    "file_idx,rxn_idx,manual_replacements_dict,expected_reactants,expected_reagents,expected_solvents,expected_catalysts,expected_products,expected_yields,expected_temperature,expected_rxn_time,expected_rxn_str,expected_names_list",
    (
        # [file_idx,rxn_idx,manual_replacements_dict,expected_reactants,expected_reagents,expected_solvents,expected_catalysts,expected_products,expected_yields,expected_temperature,expected_rxn_time,expected_rxn_str,expected_names_list],
        # [file_idx,rxn_idx,manual_replacements_dict,expected_reactants,expected_reagents,expected_solvents,expected_catalysts,expected_products,expected_yields,expected_temperature,expected_rxn_time,expected_rxn_str,expected_names_list],
    ),
)
def extract_rxn_extract(
    file_idx,
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
    expected_rxn_str,
    expected_names_list,
):
    import orderly.extract.extractor
    import orderly.data
    import orderly.extract.main

    files = orderly.extract.main.get_file_names(
        directory=orderly.data.get_path_of_test_ords(), file_ending=".pb.gz"
    )

    dataset = orderly.extract.extractor.OrdExtractor.load_data(files[file_idx])
    rxn = dataset.reactions[rxn_idx]

    (
        reactants,
        reagents,
        solvents,
        catalysts,
        products,
        yields,
        temperature,
        rxn_time,
        rxn_str,
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
    assert rxn_str == expected_rxn_str
    assert names_list == expected_names_list


@pytest.mark.parametrize(
    "trust_labelling,use_multiprocessing,name_contains_substring,inverse_substring",
    (
        [False, True, "uspto", True],
        [False, True, "uspto", False],
        [True, False, "uspto", True],
        [True, True, None, True],
    ),
)
def test_extraction_pipeline(
    tmp_path,
    trust_labelling,
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
        trust_labelling=trust_labelling,
        output_path=tmp_path,
        pickled_data_folder=pickled_data_folder,
        solvents_path=None,
        molecule_names_folder=molecule_names_folder,
        merged_molecules_file="all_molecule_names.pkl",
        use_multiprocessing=use_multiprocessing,
        name_contains_substring=name_contains_substring,
        inverse_substring=inverse_substring,
        overwrite=False,
    )

    import pandas as pd

    for extraction in pickled_data_folder.glob("*"):
        df = pd.read_pickle(extraction)

        print(df)

        # TODO tests types
        # TODO consider None vs nan

        break

    raise ValueError(0)
