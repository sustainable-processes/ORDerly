import typing
import pytest


def test_hello_world():
    assert True


def get_rxn_func() -> typing.Callable:
    from ord_schema.proto import reaction_pb2 as ord_reaction_pb2

    def get_rxn(
        file_name: str,
        rxn_idx: int,
    ) -> ord_reaction_pb2.Reaction:
        import orderly.extract.extractor
        import orderly.data
        import orderly.extract.main

        file = orderly.extract.main.get_file_names(
            directory=orderly.data.get_path_of_test_ords(),
            file_ending=f"{file_name}.pb.gz",
        )
        assert len(file) == 1
        file = file[0]

        dataset = orderly.extract.extractor.OrdExtractor.load_data(file)
        rxn = dataset.reactions[rxn_idx]
        return rxn

    return get_rxn


@pytest.mark.parametrize(
    "file_name,rxn_idx,expected_labelled_reactants,expected_labelled_reagents,expected_labelled_solvents,expected_labelled_catalysts,expected_labelled_products_from_input,expected_non_smiles_names_list_additions",
    (
        [
            "ord_dataset-00005539a1e04c809a9a78647bea649c",
            0,
            ["CC(C)N1CCNCC1", "CCOC(=O)c1cnc2cc(OCC)c(Br)cc2c1Nc1ccc(F)cc1F"],
            ["O=C([O-])[O-]", "[Cs+]", "[Cs+]"],
            [],
            [
                "c1ccc(P(c2ccccc2)c2ccc3ccccc3c2-c2c(P(c3ccccc3)c3ccccc3)ccc3ccccc23)cc1",
                "O=C(/C=C/c1ccccc1)/C=C/c1ccccc1",
                "O=C(/C=C/c1ccccc1)/C=C/c1ccccc1",
                "O=C(/C=C/c1ccccc1)/C=C/c1ccccc1",
                "[Pd]",
                "[Pd]",
            ],
            [],
            [],
        ],
        # ['ord_dataset-00005539a1e04c809a9a78647bea649c',0, ['CC(C)N1CCNCC1','CCOC(=O)c1cnc2cc(OCC)c(Br)cc2c1Nc1ccc(F)cc1F'],['O=C([O-])[O-]','[Cs+]'],[],['c1ccc(P(c2ccccc2)c2ccc3ccccc3c2-c2c(P(c3ccccc3)c3ccccc3)ccc3ccccc23)cc1', 'O=C(/C=C/c1ccccc1)/C=C/c1ccccc1', '[Pd]'],[],[]],
        # ['ord_dataset-0bb2e99daa66408fb8dbd6a0781d241c',0, expected_labelled_reactants,expected_labelled_reagents,expected_labelled_solvents,expected_labelled_catalysts,expected_labelled_products_from_input,expected_non_smiles_names_list_additions],
        # [0,1, expected_labelled_reactants,expected_labelled_reagents,expected_labelled_solvents,expected_labelled_catalysts,expected_labelled_products_from_input,expected_non_smiles_names_list_additions],
    ),
)
def test_rxn_input_extractor(
    file_name,
    rxn_idx,
    expected_labelled_reactants,
    expected_labelled_reagents,
    expected_labelled_solvents,
    expected_labelled_catalysts,
    expected_labelled_products_from_input,
    expected_non_smiles_names_list_additions,
):
    rxn = get_rxn_func()(file_name=file_name, rxn_idx=rxn_idx)

    import orderly.extract.extractor

    (
        labelled_reactants,
        labelled_reagents,
        labelled_solvents,
        labelled_catalysts,
        labelled_products_from_input,  # Daniel: I'm not sure what to do with this, it doesn't make sense for people to have put a product as an input, so this list should be empty anyway
        non_smiles_names_list_additions,
    ) = orderly.extract.extractor.OrdExtractor.rxn_input_extractor(rxn)

    assert (
        expected_labelled_reactants == labelled_reactants
    ), f"failure for {expected_labelled_reactants=}, got {labelled_reactants}"
    assert (
        expected_labelled_reagents == labelled_reagents
    ), f"failure for {expected_labelled_reagents=}, got {labelled_reagents}"
    assert (
        expected_labelled_solvents == labelled_solvents
    ), f"failure for {expected_labelled_solvents=}, got {labelled_solvents}"
    assert (
        expected_labelled_catalysts == labelled_catalysts
    ), f"failure for {expected_labelled_catalysts=}, got {labelled_catalysts}"
    assert (
        expected_labelled_products_from_input == labelled_products_from_input
    ), f"failure for {expected_labelled_products_from_input=}, got {labelled_products_from_input}"
    assert (
        expected_non_smiles_names_list_additions == non_smiles_names_list_additions
    ), f"failure for {expected_non_smiles_names_list_additions=}, got {non_smiles_names_list_additions}"


@pytest.mark.parametrize(
    "file_name,rxn_idx,expected_yields,expected_labelled_products,expected_non_smiles_names_list_additions",
    (
        [
            "ord_dataset-00005539a1e04c809a9a78647bea649c",
            0,
            ["CCOC(=O)c1cnc2cc(OCC)c(N3CCN(C(C)C)CC3)cc2c1Nc1ccc(F)cc1F"],
            [65.39],
            [],
        ],
        # [0,1, expected_yields,expected_labelled_products,expected_non_smiles_names_list_additions],
    ),
)
def test_rxn_outcomes_extractor(
    file_name,
    rxn_idx,
    expected_yields,
    expected_labelled_products,
    expected_non_smiles_names_list_additions,
):
    rxn = get_rxn_func()(file_name=file_name, rxn_idx=rxn_idx)

    import orderly.extract.extractor

    (
        yields,
        labelled_products,
        non_smiles_names_list_additions,
    ) = orderly.extract.extractor.OrdExtractor.rxn_outcomes_extractor(rxn)

    assert expected_yields == yields, f"failure for {expected_yields=} got {yields}"
    assert (
        expected_labelled_products == labelled_products
    ), f"failure for {expected_labelled_products=} got {labelled_products}"
    assert (
        expected_non_smiles_names_list_additions == non_smiles_names_list_additions
    ), f"failure for {expected_non_smiles_names_list_additions=} got {non_smiles_names_list_additions}"


@pytest.mark.parametrize(
    "file_name,rxn_idx,expected_rxn_str,expected_is_mapped",
    (
        ["ord_dataset-00005539a1e04c809a9a78647bea649c", 0, None, None],
        # [0,1, expected_rxn_str,expected_is_mapped],
    ),
)
def test_rxn_string_and_is_mapped(
    file_name,
    rxn_idx,
    expected_rxn_str,
    expected_is_mapped,
):
    rxn = get_rxn_func()(file_name=file_name, rxn_idx=rxn_idx)

    import orderly.extract.extractor

    (
        rxn_str,
        is_mapped,
    ) = orderly.extract.extractor.OrdExtractor.get_rxn_string_and_is_mapped(rxn)

    assert expected_rxn_str == rxn_str, f"failure for {expected_rxn_str=} got {rxn_str}"
    assert (
        expected_is_mapped == is_mapped
    ), f"failure for {expected_is_mapped=} got {is_mapped}"


@pytest.mark.parametrize(
    "file_name,rxn_idx,expected_rxn_info",
    (
        ["ord_dataset-00005539a1e04c809a9a78647bea649c", 0, None],
        # [0,1, expected_rxn_info],
    ),
)
def test_extract_info_from_rxn(
    file_name,
    rxn_idx,
    expected_rxn_info,
):
    rxn = get_rxn_func()(file_name=file_name, rxn_idx=rxn_idx)

    import orderly.extract.extractor

    rxn_info = orderly.extract.extractor.OrdExtractor.extract_info_from_rxn(rxn)

    assert (
        expected_rxn_info == rxn_info
    ), f"failure for {expected_rxn_info=} got {rxn_info}"


@pytest.mark.parametrize(
    "file_name,rxn_idx,expected_temperature",
    (
        ["ord_dataset-00005539a1e04c809a9a78647bea649c", 0, 110],
        # [0,1, expected_temperature],
    ),
)
def test_temperature_extractor(file_name, rxn_idx, expected_temperature):
    rxn = get_rxn_func()(file_name=file_name, rxn_idx=rxn_idx)

    import orderly.extract.extractor

    temperature = orderly.extract.extractor.OrdExtractor.temperature_extractor(rxn)

    assert (
        expected_temperature == temperature
    ), f"failure for {expected_temperature=} got {temperature}"


@pytest.mark.parametrize(
    "file_name,rxn_idx,expected_rxn_time",
    (
        ["ord_dataset-00005539a1e04c809a9a78647bea649c", 0, None],
        # [0,1, expected_rxn_time],
    ),
)
def test_time_extractor(
    file_name,
    rxn_idx,
    expected_rxn_time,
):
    rxn = get_rxn_func()(file_name=file_name, rxn_idx=rxn_idx)

    import orderly.extract.extractor

    rxn_time = orderly.extract.extractor.OrdExtractor.rxn_time_extractor(rxn)

    assert (
        expected_rxn_time == rxn_time
    ), f"failure for {expected_rxn_time=} got {rxn_time}"


@pytest.mark.parametrize(
    "rxn_str_agents,labelled_catalysts,labelled_solvents,labelled_reagents,metals,solvents_set,expected_agents,expected_solvents",
    (
        [
            None,
            [
                "c1ccc(P(c2ccccc2)c2ccc3ccccc3c2-c2c(P(c3ccccc3)c3ccccc3)ccc3ccccc23)cc1",
                "O=C(/C=C/c1ccccc1)/C=C/c1ccccc1",
                "[Pd]",
            ],
            [],
            ["O=C([O-])[O-]", "[Cs+]"],
            None,
            None,
            [
                "[Cs+]",
                "[Pd]",
                "O=C(/C=C/c1ccccc1)/C=C/c1ccccc1",
                "O=C([O-])[O-]",
                "c1ccc(P(c2ccccc2)c2ccc3ccccc3c2-c2c(P(c3ccccc3)c3ccccc3)ccc3ccccc23)cc1",
            ],
            [],
        ],
        # [rxn_str_agents,labelled_catalysts,labelled_solvents,labelled_reagents,metals,solvents_set,expected_agents,expected_solvents],
        # [rxn_str_agents,labelled_catalysts,labelled_solvents,labelled_reagents,metals,solvents_set,expected_agents,expected_solvents],
    ),
)
def test_merge_to_agents(
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

    if metals is None:
        metals = orderly.extract.defaults.get_metals_list()
    if solvents_set is None:
        solvents_set = orderly.extract.defaults.get_solvents_set()

    agents, solvents = orderly.extract.extractor.OrdExtractor.merge_to_agents(
        rxn_str_agents,
        labelled_catalysts,
        labelled_solvents,
        labelled_reagents,
        metals,
        solvents_set,
    )

    assert expected_agents == agents, f"failure for {expected_agents=} got {agents}"
    assert (
        expected_solvents == solvents
    ), f"failure for {expected_solvents=} got {solvents}"


@pytest.mark.parametrize(
    "rxn_str_products,labelled_products,input_yields,expected_products,expected_yields",
    (
        [
            [],
            ["CCOC(=O)c1cnc2cc(OCC)c(N3CCN(C(C)C)CC3)cc2c1Nc1ccc(F)cc1F"],
            [65.39],
            ["CCOC(=O)c1cnc2cc(OCC)c(N3CCN(C(C)C)CC3)cc2c1Nc1ccc(F)cc1F"],
            [65.39],
        ],
        # [rxn_str_products,labelled_products,input_yields,expected_products,expected_yields],
    ),
)
def test_match_yield_with_product(
    rxn_str_products,
    input_yields,
    labelled_products,
    expected_yields,
    expected_products,
):
    import orderly.extract.extractor

    products, yields = orderly.extract.extractor.OrdExtractor.match_yield_with_product(
        rxn_str_products, labelled_products, input_yields
    )

    assert (
        expected_products == products
    ), f"failure for {expected_products=} got {products}"
    assert expected_yields == yields, f"failure for {expected_yields=} got {yields}"


@pytest.mark.parametrize(
    "file_name,rxn_idx,manual_replacements_dict,expected_reactants,expected_reagents,expected_solvents,expected_catalysts,expected_products,expected_yields,expected_temperature,expected_rxn_time,expected_rxn_str,expected_names_list",
    (
        # [file_name,rxn_idx,manual_replacements_dict,expected_reactants,expected_reagents,expected_solvents,expected_catalysts,expected_products,expected_yields,expected_temperature,expected_rxn_time,expected_rxn_str,expected_names_list],
        # [file_name,rxn_idx,manual_replacements_dict,expected_reactants,expected_reagents,expected_solvents,expected_catalysts,expected_products,expected_yields,expected_temperature,expected_rxn_time,expected_rxn_str,expected_names_list],
    ),
)
def handle_reaction_object(
    file_name,
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
    rxn = get_rxn_func()(file_name=file_name, rxn_idx=rxn_idx)

    import orderly.extract.extractor

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
    assert (
        reactants == expected_reactants
    ), f"failure for {expected_reactants=} got {reactants}"
    assert (
        reagents == expected_reagents
    ), f"failure for {expected_reagents=} got {reagents}"
    assert (
        solvents == expected_solvents
    ), f"failure for {expected_solvents=} got {solvents}"
    assert (
        catalysts == expected_catalysts
    ), f"failure for {expected_catalysts=} got {catalysts}"
    assert (
        products == expected_products
    ), f"failure for {expected_products=} got {products}"
    assert yields == expected_yields, f"failure for {expected_yields=} got {yields}"
    assert (
        temperature == expected_temperature
    ), f"failure for {expected_temperature=} got {temperature}"
    assert (
        rxn_time == expected_rxn_time
    ), f"failure for {expected_rxn_time=} got {rxn_time}"
    assert rxn_str == expected_rxn_str, f"failure for {expected_rxn_str=} got {rxn_str}"
    assert (
        names_list == expected_names_list
    ), f"failure for {expected_names_list=} got {names_list}"


@pytest.mark.parametrize(
    "trust_labelling,use_multiprocessing,name_contains_substring,inverse_substring",
    (
        [False, True, "uspto", True],
        [False, True, "uspto", False],
        [True, False, "uspto", True],
        [True, True, None, True],
    ),
)
def extraction_pipeline(
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
        # TODO should is_mapped be False or None if there is no reaction string?

        break
