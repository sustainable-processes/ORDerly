import typing
import pytest
import pathlib

REPETITIONS = 3
SLOW_REPETITIONS = 1


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
        file = pathlib.Path(file[0])

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
        [
            "ord_dataset-0b70410902ae4139bd5d334881938f69",
            0,
            [
                "SCc1ccccc1",
                "[H-]",
                "[Na+]",
                "O=[N+]([O-])c1ccc(Oc2ccc(C(F)(F)F)cc2Cl)cc1[N+](=O)[O-]",
            ],
            [],
            ["C1CCOC1", "C1CCOC1"],
            [],
            [],
            [],
        ],
        [
            "ord_dataset-0bb2e99daa66408fb8dbd6a0781d241c",
            0,
            [
                "[Na+]",
                "[Na+]",
                "[Na+]",
                "[Na+]",
                "[Na+]",
                "[Na+]",
                "[Na+]",
                "[Na+]",
                "[Na+]",
                "[Na+]",
                "[Na+]",
                "[Na+]",
                "[O-]B([O-])[O-]",
                "[O-]B([O-])[O-]",
                "[O-]B([O-])[O-]",
                "[O-]B([O-])[O-]",
                "35(Na2O)",
            ],
            [],
            ["O"],
            [],
            [],
            ["35(Na2O)"],
        ],
        [
            "ord_dataset-0bf72e95d80743729fdbb8b57a4bc0c6",
            0,
            ["CC1(C)C2CCC(=O)C1C2", "C1CCNC1", "Cc1ccc(S(=O)(=O)O)cc1", "O"],
            [],
            ["c1ccccc1"],
            [],
            [],
            [],
        ],
    ),
)
@pytest.mark.parametrize("execution_number", range(REPETITIONS))
def test_rxn_input_extractor(
    execution_number,
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

    assert sorted(expected_labelled_reactants) == sorted(
        labelled_reactants
    ), f"failure for {sorted(expected_labelled_reactants)=}, got {sorted(labelled_reactants)}"  # TODO unsure why we have random ordering on ubuntu
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
    "file_name,rxn_idx,expected_labelled_products,expected_yields,expected_non_smiles_names_list_additions",
    (
        [
            "ord_dataset-00005539a1e04c809a9a78647bea649c",
            0,
            ["CCOC(=O)c1cnc2cc(OCC)c(N3CCN(C(C)C)CC3)cc2c1Nc1ccc(F)cc1F"],
            [65.39],
            [],
        ],
        [
            "ord_dataset-0b70410902ae4139bd5d334881938f69",
            0,
            ["O=[N+]([O-])c1ccc(Oc2ccc(C(F)(F)F)cc2Cl)cc1SCc1ccccc1"],
            [None],
            [],
        ],
        [
            "ord_dataset-0bb2e99daa66408fb8dbd6a0781d241c",
            0,
            ["[O-]B1OB2OB([O-])OB(O1)O2", "[Na+]", "[Na+]"],
            [None, None, None],
            [],
        ],
        [
            "ord_dataset-0bf72e95d80743729fdbb8b57a4bc0c6",
            0,
            ["CC1(C)C2CC=C(N3CCCC3)C1C2"],
            [95.0],
            [],
        ],
    ),
)
@pytest.mark.parametrize("execution_number", range(REPETITIONS))
def test_rxn_outcomes_extractor(
    execution_number,
    file_name,
    rxn_idx,
    expected_labelled_products,
    expected_yields,
    expected_non_smiles_names_list_additions,
):
    rxn = get_rxn_func()(file_name=file_name, rxn_idx=rxn_idx)

    import orderly.extract.extractor

    (
        labelled_products,
        yields,
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
        [
            "ord_dataset-0b70410902ae4139bd5d334881938f69",
            0,
            "[CH2:1]([SH:8])[C:2]1[CH:7]=[CH:6][CH:5]=[CH:4][CH:3]=1.[H-].[Na+].[Cl:11][C:12]1[CH:30]=[C:29]([C:31]([F:34])([F:33])[F:32])[CH:28]=[CH:27][C:13]=1[O:14][C:15]1[CH:20]=[CH:19][C:18]([N+:21]([O-:23])=[O:22])=[C:17]([N+]([O-])=O)[CH:16]=1>O1CCCC1>[CH2:1]([S:8][C:17]1[CH:16]=[C:15]([O:14][C:13]2[CH:27]=[CH:28][C:29]([C:31]([F:34])([F:32])[F:33])=[CH:30][C:12]=2[Cl:11])[CH:20]=[CH:19][C:18]=1[N+:21]([O-:23])=[O:22])[C:2]1[CH:7]=[CH:6][CH:5]=[CH:4][CH:3]=1",
            True,
        ],
        [
            "ord_dataset-0bb2e99daa66408fb8dbd6a0781d241c",
            0,
            "[B:1]([O-:4])([O-:3])[O-:2].[B:5]([O-:8])([O-:7])[O-:6].[B:9]([O-:12])([O-])[O-].[B:13]([O-])([O-])[O-].[Na+:17].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+]>O>[B:1]1([O-:4])[O:3][B:13]2[O:12][B:9]([O:6][B:5]([O-:8])[O:7]2)[O:2]1.[Na+:17].[Na+:17]",
            True,
        ],
        [
            "ord_dataset-0bf72e95d80743729fdbb8b57a4bc0c6",
            0,
            "[CH3:1][C:2]1([CH3:10])[CH:8]2[CH2:9][CH:3]1[CH2:4][CH2:5][C:6]2=O.[NH:11]1[CH2:15][CH2:14][CH2:13][CH2:12]1.C1(C)C=CC(S(O)(=O)=O)=CC=1.O>C1C=CC=CC=1>[CH3:1][C:2]1([CH3:10])[CH:8]2[CH2:9][CH:3]1[CH2:4][CH:5]=[C:6]2[N:11]1[CH2:15][CH2:14][CH2:13][CH2:12]1",
            True,
        ],
    ),
)
@pytest.mark.parametrize("execution_number", range(REPETITIONS))
def test_rxn_string_and_is_mapped(
    execution_number,
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
    "file_name,rxn_idx,rxn_overwrite,expected_rxn_str_reactants,expected_rxn_str_agents,expected_rxn_str_products,expected_rxn_str,expected_non_smiles_names_list_additions,expected_none",
    (
        [
            "ord_dataset-00005539a1e04c809a9a78647bea649c",
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            True,
        ],
        [
            "ord_dataset-0b70410902ae4139bd5d334881938f69",
            0,
            None,
            [
                "O=[N+]([O-])c1ccc(Oc2ccc(C(F)(F)F)cc2Cl)cc1[N+](=O)[O-]",
                "SCc1ccccc1",
            ],
            [
                "C1CCOC1",
                "[H-]",
                "[Na+]",
            ],
            ["O=[N+]([O-])c1ccc(Oc2ccc(C(F)(F)F)cc2Cl)cc1SCc1ccccc1"],
            "[CH2:1]([SH:8])[C:2]1[CH:7]=[CH:6][CH:5]=[CH:4][CH:3]=1.[H-].[Na+].[Cl:11][C:12]1[CH:30]=[C:29]([C:31]([F:34])([F:33])[F:32])[CH:28]=[CH:27][C:13]=1[O:14][C:15]1[CH:20]=[CH:19][C:18]([N+:21]([O-:23])=[O:22])=[C:17]([N+]([O-])=O)[CH:16]=1>O1CCCC1>[CH2:1]([S:8][C:17]1[CH:16]=[C:15]([O:14][C:13]2[CH:27]=[CH:28][C:29]([C:31]([F:34])([F:32])[F:33])=[CH:30][C:12]=2[Cl:11])[CH:20]=[CH:19][C:18]=1[N+:21]([O-:23])=[O:22])[C:2]1[CH:7]=[CH:6][CH:5]=[CH:4][CH:3]=1",
            [],
            False,
        ],
        [
            "ord_dataset-0bb2e99daa66408fb8dbd6a0781d241c",
            0,
            None,
            [
                "[O-]B([O-])[O-]",
            ],
            [
                "O",
                "[Na+]",
            ],
            [
                "[O-]B1OB2OB([O-])OB(O1)O2",
            ],
            "[B:1]([O-:4])([O-:3])[O-:2].[B:5]([O-:8])([O-:7])[O-:6].[B:9]([O-:12])([O-])[O-].[B:13]([O-])([O-])[O-].[Na+:17].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+]>O>[B:1]1([O-:4])[O:3][B:13]2[O:12][B:9]([O:6][B:5]([O-:8])[O:7]2)[O:2]1.[Na+:17].[Na+:17]",
            [],
            False,
        ],
        [
            "ord_dataset-0bf72e95d80743729fdbb8b57a4bc0c6",
            0,
            None,
            [
                "C1CCNC1",
                "CC1(C)C2CCC(=O)C1C2",
            ],
            ["Cc1ccc(S(=O)(=O)O)cc1", "O", "c1ccccc1"],
            ["CC1(C)C2CC=C(N3CCCC3)C1C2"],
            "[CH3:1][C:2]1([CH3:10])[CH:8]2[CH2:9][CH:3]1[CH2:4][CH2:5][C:6]2=O.[NH:11]1[CH2:15][CH2:14][CH2:13][CH2:12]1.C1(C)C=CC(S(O)(=O)=O)=CC=1.O>C1C=CC=CC=1>[CH3:1][C:2]1([CH3:10])[CH:8]2[CH2:9][CH:3]1[CH2:4][CH:5]=[C:6]2[N:11]1[CH2:15][CH2:14][CH2:13][CH2:12]1",
            [],
            False,
        ],
        # Test case where reaction with only 1 >
        pytest.param(
            "ord_dataset-0bf72e95d80743729fdbb8b57a4bc0c6",
            0,
            "CC.C>CCC",
            ["CC", "C"],
            [],
            ["CCC"],
            "CC.C>CCC",
            [],
            False,
            marks=pytest.mark.xfail(
                reason="ValueError: not enough values to unpack (expected 3, got 2)"
            ),
        ),
        # There's no point in trying to test whether the the rxn.identifiers[0].value = None because the schema doesn't allow that overwrite to happen!
    ),
)
@pytest.mark.parametrize("execution_number", range(REPETITIONS))
def test_extract_info_from_rxn(
    execution_number,
    file_name,
    rxn_idx,
    rxn_overwrite,
    expected_rxn_str_reactants,
    expected_rxn_str_agents,
    expected_rxn_str_products,
    expected_rxn_str,
    expected_non_smiles_names_list_additions,
    expected_none,
):
    rxn = get_rxn_func()(file_name=file_name, rxn_idx=rxn_idx)

    if rxn_overwrite is not None:
        rxn.identifiers[0].value = rxn_overwrite

    import orderly.extract.extractor

    rxn_info = orderly.extract.extractor.OrdExtractor.extract_info_from_rxn(rxn)
    if expected_none:
        assert rxn_info is None, f"expected a none but got {rxn_info=}"
        return
    else:
        assert rxn_info is not None, f"did not expect a none but got one {rxn_info=}"
    (
        rxn_str_reactants,
        rxn_str_agents,
        rxn_str_products,
        rxn_str,
        non_smiles_names_list_additions,
    ) = rxn_info

    assert expected_rxn_str_reactants == rxn_str_reactants
    assert expected_rxn_str_agents == rxn_str_agents
    assert expected_rxn_str_products == rxn_str_products
    assert expected_rxn_str == rxn_str
    assert expected_non_smiles_names_list_additions == non_smiles_names_list_additions


@pytest.mark.parametrize(
    "file_name,rxn_idx,expected_temperature",
    (
        [
            "ord_dataset-00005539a1e04c809a9a78647bea649c",
            0,
            110.0,
        ],
        ["ord_dataset-0b70410902ae4139bd5d334881938f69", 0, None],
        ["ord_dataset-0bb2e99daa66408fb8dbd6a0781d241c", 0, 1100.0],
        ["ord_dataset-0bf72e95d80743729fdbb8b57a4bc0c6", 0, None],
    ),
)
@pytest.mark.parametrize("execution_number", range(REPETITIONS))
def test_temperature_extractor(
    execution_number, file_name, rxn_idx, expected_temperature
):
    rxn = get_rxn_func()(file_name=file_name, rxn_idx=rxn_idx)

    import orderly.extract.extractor

    temperature = orderly.extract.extractor.OrdExtractor.temperature_extractor(rxn)

    assert (
        expected_temperature == temperature
    ), f"failure for {expected_temperature=} got {temperature}"
    if temperature is not None:
        assert isinstance(
            temperature, float
        ), f"expected a float but got {temperature=}"


@pytest.mark.parametrize(
    "file_name,rxn_idx,expected_rxn_time",
    (
        ["ord_dataset-00005539a1e04c809a9a78647bea649c", 0, None],
        ["ord_dataset-0b70410902ae4139bd5d334881938f69", 0, None],
        ["ord_dataset-0bb2e99daa66408fb8dbd6a0781d241c", 0, 0.17],
        ["ord_dataset-0bf72e95d80743729fdbb8b57a4bc0c6", 0, None],
    ),
)
@pytest.mark.parametrize("execution_number", range(REPETITIONS))
def test_time_extractor(
    execution_number,
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

    if rxn_time is not None:
        assert isinstance(rxn_time, float), f"expected a float but got {rxn_time=}"


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
        [["C1CCOC1"], None, ["C1CCOC1", "C1CCOC1"], None, None, None, [], ["C1CCOC1"]],
        [["O"], [], ["O"], [], None, None, [], ["O"]],
        [
            ["c1ccccc1", "Cc1ccc(S(=O)(=O)O)cc1", "O"],
            [],
            ["c1ccccc1"],
            [],
            None,
            None,
            ["Cc1ccc(S(=O)(=O)O)cc1"],
            ["O", "c1ccccc1"],
        ],
        # Made up test cases:
        [
            ["c1ccccc1", "Cc1ccc(S(=O)(=O)O)cc1", "O", None],
            ["[Pd]"],
            ["O", "CCO"],
            ["O=C([O-])[O-]"],
            None,
            None,
            ["[Pd]", "Cc1ccc(S(=O)(=O)O)cc1", "O=C([O-])[O-]"],
            ["CCO", "O", "c1ccccc1"],
        ],
        pytest.param(
            ["c1ccccc1", "Cc1ccc(S(=O)(=O)O)cc1", "O"],
            ["[Pd]"],
            ["O", "CCO"],
            ["O=C([O-])[O-]"],
            None,
            None,
            ["c1ccccc1", "Cc1ccc(S(=O)(=O)O)cc1", "O"],
            ["O", "CCO"],
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            ["c1ccccc1", "Cc1ccc(S(=O)(=O)O)cc1", "O"],
            ["[Pd]"],
            ["O", "CCO"],
            ["O=C([O-])[O-]"],
            None,
            None,
            ["[Pd]", "Cc1ccc(S(=O)(=O)O)cc1", "O=C([O-])[O-]", "O", "CCO", "c1ccccc1"],
            [],
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            ["c1ccccc1", "Cc1ccc(S(=O)(=O)O)cc1", "O"],
            ["[Pd]"],
            ["O", "CCO"],
            ["O=C([O-])[O-]"],
            None,
            None,
            ["[Pd]", "Cc1ccc(S(=O)(=O)O)cc1", "O=C([O-])[O-]"],
            ["O", "O", "CCO", "c1ccccc1"],
            marks=pytest.mark.xfail,
        ),
    ),
)
@pytest.mark.parametrize("execution_number", range(REPETITIONS))
def test_merge_to_agents(
    execution_number,
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
        [
            ["O=[N+]([O-])c1ccc(Oc2ccc(C(F)(F)F)cc2Cl)cc1SCc1ccccc1"],
            ["O=[N+]([O-])c1ccc(Oc2ccc(C(F)(F)F)cc2Cl)cc1SCc1ccccc1"],
            None,
            ["O=[N+]([O-])c1ccc(Oc2ccc(C(F)(F)F)cc2Cl)cc1SCc1ccccc1"],
            None,
        ],
        [
            ["[Na+]", "[Na+]", "[O-]B1OB2OB([O-])OB(O1)O2"],
            ["[Na+]", "[Na+]", "[O-]B1OB2OB([O-])OB(O1)O2"],
            [None, None, None],
            ["[Na+]", "[Na+]", "[O-]B1OB2OB([O-])OB(O1)O2"],
            [None, None, None],
        ],
        pytest.param(
            ["[Na+]", "[Na+]", "[O-]B1OB2OB([O-])OB(O1)O2"],
            ["[Na+]", "[Na+]", "[O-]B1OB2OB([O-])OB(O1)O2"],
            [None],
            ["[Na+]", "[Na+]", "[O-]B1OB2OB([O-])OB(O1)O2"],
            [None],
            marks=pytest.mark.xfail(reason="IndexError: list index out of range"),
        ),
        [
            ["O=[N+]([O-])c1ccc(Oc2ccc(C(F)(F)F)cc2Cl)cc1SCc1ccccc1"],
            ["[Na+]", "[Na+]", "[O-]B1OB2OB([O-])OB(O1)O2"],
            [None, None, None],
            ["O=[N+]([O-])c1ccc(Oc2ccc(C(F)(F)F)cc2Cl)cc1SCc1ccccc1"],
            [None],
        ],
        [
            ["CC1(C)C2CC=C(N3CCCC3)C1C2"],
            ["CC1(C)C2CC=C(N3CCCC3)C1C2"],
            [95.0],
            ["CC1(C)C2CC=C(N3CCCC3)C1C2"],
            [95.0],
        ],
    ),
)
@pytest.mark.parametrize("execution_number", range(REPETITIONS))
def test_match_yield_with_product(
    execution_number,
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

    assert (
        expected_products == products
    ), f"failure for {expected_products=} got {products}"
    assert expected_yields == yields, f"failure for {expected_yields=} got {yields}"


@pytest.mark.parametrize(
    "file_name,rxn_idx,manual_replacements_dict,trust_labelling,expected_reactants, expected_agents, expected_reagents,expected_solvents,expected_catalysts,expected_products,expected_yields,expected_temperature,expected_rxn_time,expected_rxn_str, expected_procedure_details, expected_names_list",
    (
        [
            "ord_dataset-00005539a1e04c809a9a78647bea649c",
            0,
            {},
            False,
            ["CC(C)N1CCNCC1", "CCOC(=O)c1cnc2cc(OCC)c(Br)cc2c1Nc1ccc(F)cc1F"],
            [
                "[Cs+]",
                "[Pd]",
                "O=C(/C=C/c1ccccc1)/C=C/c1ccccc1",
                "O=C([O-])[O-]",
                "c1ccc(P(c2ccccc2)c2ccc3ccccc3c2-c2c(P(c3ccccc3)c3ccccc3)ccc3ccccc23)cc1",
            ],
            [],
            [],
            [],
            ["CCOC(=O)c1cnc2cc(OCC)c(N3CCN(C(C)C)CC3)cc2c1Nc1ccc(F)cc1F"],
            [65.39],
            110.0,
            None,
            None,
            "To a solution of ethyl 6-bromo-4-(2,4-difluorophenylamino)-7-ethoxyquinoline-3-carboxylate (400 mg, 0.89 mmol) and 1-(Isopropyl)piperazine (254 µl, 1.77 mmol) in dioxane was added cesium carbonate (722 mg, 2.22 mmol), tris(dibenzylideneacetone)dipalladium(0) (40.6 mg, 0.04 mmol) and rac-2,2'-Bis(diphenylphosphino)-1,1'-binaphthyl (55.2 mg, 0.09 mmol). Reaction vessel in oil bath set to 110 °C. 11am  After 5 hours, MS shows product (major peak 499), and SM (minor peak 453).  o/n, MS shows product peak. Reaction cooled, concentrated onto silica, and purified on ISCO. 40g column, 1:1 EA:Hex, then 100% EA.  289mg yellow solid. NMR (EN00180-62-1) supports product, but some oxidised BINAP impurity (LCMS 655).  ",
            [],
        ],
        [
            "ord_dataset-0b70410902ae4139bd5d334881938f69",
            0,
            {},
            False,
            [
                "SCc1ccccc1",
                "O=[N+]([O-])c1ccc(Oc2ccc(C(F)(F)F)cc2Cl)cc1[N+](=O)[O-]",
            ],
            ["[Na+]", "[H-]"],
            [],
            ["C1CCOC1"],
            [],
            ["O=[N+]([O-])c1ccc(Oc2ccc(C(F)(F)F)cc2Cl)cc1SCc1ccccc1"],
            [None],
            None,
            None,
            "[CH2:1]([SH:8])[C:2]1[CH:7]=[CH:6][CH:5]=[CH:4][CH:3]=1.[H-].[Na+].[Cl:11][C:12]1[CH:30]=[C:29]([C:31]([F:34])([F:33])[F:32])[CH:28]=[CH:27][C:13]=1[O:14][C:15]1[CH:20]=[CH:19][C:18]([N+:21]([O-:23])=[O:22])=[C:17]([N+]([O-])=O)[CH:16]=1>O1CCCC1>[CH2:1]([S:8][C:17]1[CH:16]=[C:15]([O:14][C:13]2[CH:27]=[CH:28][C:29]([C:31]([F:34])([F:32])[F:33])=[CH:30][C:12]=2[Cl:11])[CH:20]=[CH:19][C:18]=1[N+:21]([O-:23])=[O:22])[C:2]1[CH:7]=[CH:6][CH:5]=[CH:4][CH:3]=1",
            "1.7 g of benzyl mercaptan was dissolved in dry tetrahydrofuran and 0.5 g of sodium hydride added with stirring under dry nitrogen. The reaction mixture was stirred under reflux for 30 minutes, and a solution of 5 g of 1A dissolved in 25 ml of dry tetrahydrofuran was added dropwise. Reaction occurred rapidly, and the product was chromatographically purified to give 2-benzylthio-4-(2-chloro-4-trifluoromethylphenoxy)nitrobenzene (1B) as a yellow oil.",
            [],
        ],
        [
            "ord_dataset-0bb2e99daa66408fb8dbd6a0781d241c",
            0,
            {},
            False,
            ["[O-]B([O-])[O-]"],
            ["[Na+]"],
            [],
            ["O"],
            [],
            ["[O-]B1OB2OB([O-])OB(O1)O2"],
            [None],
            1100.0,
            0.17,
            "[B:1]([O-:4])([O-:3])[O-:2].[B:5]([O-:8])([O-:7])[O-:6].[B:9]([O-:12])([O-])[O-].[B:13]([O-])([O-])[O-].[Na+:17].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+]>O>[B:1]1([O-:4])[O:3][B:13]2[O:12][B:9]([O:6][B:5]([O-:8])[O:7]2)[O:2]1.[Na+:17].[Na+:17]",
            "Sodium tetraborate (Na2B4O7.10H2O), analyzed reagent was dried overnight at 150° C, mixed with the appropriate quantity of dopant ions and homogenized in an electric homogenizer (vibrator) during 10 minutes. The material was then transferred to a platinum crucible and heated at 1100° C for at least 30 minutes, until a clear transparent solution was obtained. The glass matrix loses water and the composition of the matrix is after the heating 35(Na2O).65(B2O3). A drop of the hot melt was allowed to fall directly onto a clean white glazed ceramic surface, into the center of a space ring of 1 mm thickness, and pressed with a second ceramic tile to produce a glass disk of 1 mm thickness and an approximate diameter of 12 mm. The glass is transparent in the ultraviolet and in the visible part of the spectrum.",
            [],
        ],
        [
            "ord_dataset-0bb2e99daa66408fb8dbd6a0781d241c",
            0,
            {},
            True,
            [
                "[Na+]",
                "[Na+]",
                "[Na+]",
                "[Na+]",
                "[Na+]",
                "[Na+]",
                "[Na+]",
                "[Na+]",
                "[Na+]",
                "[Na+]",
                "[Na+]",
                "[Na+]",
                "[O-]B([O-])[O-]",
                "[O-]B([O-])[O-]",
                "[O-]B([O-])[O-]",
                "[O-]B([O-])[O-]",
                "35(Na2O)",
            ],
            [],
            [],
            ["O"],
            [],
            [
                "[O-]B1OB2OB([O-])OB(O1)O2",
                "[Na+]",
                "[Na+]",
            ],
            [None, None, None],
            1100.0,
            0.17,
            "[B:1]([O-:4])([O-:3])[O-:2].[B:5]([O-:8])([O-:7])[O-:6].[B:9]([O-:12])([O-])[O-].[B:13]([O-])([O-])[O-].[Na+:17].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+]>O>[B:1]1([O-:4])[O:3][B:13]2[O:12][B:9]([O:6][B:5]([O-:8])[O:7]2)[O:2]1.[Na+:17].[Na+:17]",
            "Sodium tetraborate (Na2B4O7.10H2O), analyzed reagent was dried overnight at 150° C, mixed with the appropriate quantity of dopant ions and homogenized in an electric homogenizer (vibrator) during 10 minutes. The material was then transferred to a platinum crucible and heated at 1100° C for at least 30 minutes, until a clear transparent solution was obtained. The glass matrix loses water and the composition of the matrix is after the heating 35(Na2O).65(B2O3). A drop of the hot melt was allowed to fall directly onto a clean white glazed ceramic surface, into the center of a space ring of 1 mm thickness, and pressed with a second ceramic tile to produce a glass disk of 1 mm thickness and an approximate diameter of 12 mm. The glass is transparent in the ultraviolet and in the visible part of the spectrum.",
            ["35(Na2O)"],
        ],
        [
            "ord_dataset-0bf72e95d80743729fdbb8b57a4bc0c6",
            0,
            {},
            False,
            [
                "CC1(C)C2CCC(=O)C1C2",
                "C1CCNC1",
            ],
            [
                "Cc1ccc(S(=O)(=O)O)cc1",
            ],
            [],
            [
                "O",
                "c1ccccc1",
            ],
            [],
            ["CC1(C)C2CC=C(N3CCCC3)C1C2"],
            [95.0],
            None,
            None,
            "[CH3:1][C:2]1([CH3:10])[CH:8]2[CH2:9][CH:3]1[CH2:4][CH2:5][C:6]2=O.[NH:11]1[CH2:15][CH2:14][CH2:13][CH2:12]1.C1(C)C=CC(S(O)(=O)=O)=CC=1.O>C1C=CC=CC=1>[CH3:1][C:2]1([CH3:10])[CH:8]2[CH2:9][CH:3]1[CH2:4][CH:5]=[C:6]2[N:11]1[CH2:15][CH2:14][CH2:13][CH2:12]1",
            "A solution of 30 g of nopinone ([α]D20 =+39.90; c=8 in ethanol), 29 of pyrrolidine and 0.4 g of p-toluenesulfonic acid in 150 ml anhydrous benzene was heated at reflux for 40 h under nitrogen atmosphere in a vessel fitted with a water separator. After evaporation of the solvent and distillation of the residue, there were obtained 39.5 g (95% yield) of 1-(6,6-dimethylnorpin-2-en-2-yl)-pyrrolidine having b.p. 117°-118° C./10 Torr.",
            [],
        ],
        [
            "ord_dataset-0bf72e95d80743729fdbb8b57a4bc0c6",
            0,
            {},
            True,
            [
                "C1CCNC1",
                "CC1(C)C2CCC(=O)C1C2",
                "O",
                "Cc1ccc(S(=O)(=O)O)cc1",
            ],
            [],
            [],
            ["c1ccccc1"],
            [],
            ["CC1(C)C2CC=C(N3CCCC3)C1C2"],
            [95.0],
            None,
            None,
            "[CH3:1][C:2]1([CH3:10])[CH:8]2[CH2:9][CH:3]1[CH2:4][CH2:5][C:6]2=O.[NH:11]1[CH2:15][CH2:14][CH2:13][CH2:12]1.C1(C)C=CC(S(O)(=O)=O)=CC=1.O>C1C=CC=CC=1>[CH3:1][C:2]1([CH3:10])[CH:8]2[CH2:9][CH:3]1[CH2:4][CH:5]=[C:6]2[N:11]1[CH2:15][CH2:14][CH2:13][CH2:12]1",
            "A solution of 30 g of nopinone ([α]D20 =+39.90; c=8 in ethanol), 29 of pyrrolidine and 0.4 g of p-toluenesulfonic acid in 150 ml anhydrous benzene was heated at reflux for 40 h under nitrogen atmosphere in a vessel fitted with a water separator. After evaporation of the solvent and distillation of the residue, there were obtained 39.5 g (95% yield) of 1-(6,6-dimethylnorpin-2-en-2-yl)-pyrrolidine having b.p. 117°-118° C./10 Torr.",
            [],
        ],
    ),
)
# TODO: add test to check whether manual_replacements_dict is used properly
@pytest.mark.parametrize("execution_number", range(REPETITIONS))
def test_handle_reaction_object(
    execution_number,
    file_name,
    rxn_idx,
    manual_replacements_dict,
    trust_labelling,
    expected_reactants,
    expected_agents,
    expected_reagents,
    expected_solvents,
    expected_catalysts,
    expected_products,
    expected_yields,
    expected_temperature,
    expected_rxn_time,
    expected_rxn_str,
    expected_procedure_details,
    expected_names_list,
):
    import orderly.extract.extractor

    rxn = get_rxn_func()(file_name=file_name, rxn_idx=rxn_idx)
    if len(manual_replacements_dict) == 0:
        manual_replacements_dict = orderly.extract.main.get_manual_replacements_dict(
            solvents_path=None
        )

    metals = orderly.extract.defaults.get_metals_list()
    solvents_set = orderly.extract.defaults.get_solvents_set()

    (
        reactants,
        agents,
        reagents,
        solvents,
        catalysts,
        products,
        yields,
        temperature,
        rxn_time,
        rxn_str,
        procedure_details,
        names_list,
    ) = orderly.extract.extractor.OrdExtractor.handle_reaction_object(
        rxn, manual_replacements_dict, solvents_set, metals, trust_labelling
    )

    def clean_string(s):
        import string

        printable = set(string.printable)
        return "".join(filter(lambda x: x in printable, s))

    clean_procedure_details = clean_string(procedure_details)
    clean_expected_procedure_details = clean_string(expected_procedure_details)

    assert sorted(reactants) == sorted(
        expected_reactants
    ), f"failure for {sorted(expected_reactants)=} got {sorted(reactants)}"  # TODO unsure why we have random ordering on ubuntu
    assert agents == expected_agents, f"failure for {expected_agents=} got {agents}"
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
    assert (
        clean_procedure_details == clean_expected_procedure_details
    ), f"failure for {clean_expected_procedure_details=} got {clean_procedure_details}"


@pytest.mark.parametrize(
    "smiles,is_mapped,expected_canonical_smiles",
    (
        ["teststring", True, None],
        ["teststring", False, None],
        ["c1ccccc1", False, "c1ccccc1"],
        [
            "[CH2:1]([S:8][C:9]1[CH:14]=[C:13]([O:15][C:16]2[CH:21]=[CH:20][C:19]([C:22]([F:25])([F:24])[F:23])=[CH:18][C:17]=2[Cl:26])[CH:12]=[CH:11][C:10]=1[N+:27]([O-])=O)[C:2]1[CH:7]=[CH:6][CH:5]=[CH:4][CH:3]=1",
            True,
            "O=[N+]([O-])c1ccc(Oc2ccc(C(F)(F)F)cc2Cl)cc1SCc1ccccc1",
        ],
        ["P([O-])(O)O", False, "[O-]P(O)O"],
        [
            "[CC(C)(C)[P]([Pd][P](C(C)(C)C)(C(C)(C)C)C(C)(C)C)(C(C)(C)C)C(C)(C)C]",
            False,
            "CC(C)(C)[P]([Pd][P](C(C)(C)C)(C(C)(C)C)C(C)(C)C)(C(C)(C)C)C(C)(C)C",
        ],
        [
            "[CC(C)(C)[P]([Pd][P](C(C)(C)C)(C(C)(C)C)C(C)(C)C)(C(C)(C)C)C(C)(C)C]",
            True,
            "CC(C)(C)[P]([Pd][P](C(C)(C)C)(C(C)(C)C)C(C)(C)C)(C(C)(C)C)C(C)(C)C",
        ],
    ),
)
@pytest.mark.parametrize("execution_number", range(REPETITIONS))
def test_canonicalisation(
    execution_number,
    smiles,
    is_mapped,
    expected_canonical_smiles,
):
    from orderly.extract.canonicalise import get_canonicalised_smiles

    canonical_smiles = get_canonicalised_smiles(smiles, is_mapped)

    assert (
        expected_canonical_smiles == canonical_smiles
    ), f"failure for {expected_canonical_smiles=} got {canonical_smiles}"


@pytest.mark.parametrize(
    "trust_labelling,use_multiprocessing,name_contains_substring,inverse_substring",
    (
        [False, True, "uspto", True],
        [
            False,
            True,
            "uspto",
            False,
        ],
        [True, False, "uspto", True],
        [True, True, None, True],
    ),
)
@pytest.mark.parametrize("execution_number", range(SLOW_REPETITIONS))
def test_extraction_pipeline(
    execution_number,
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
        data_path=orderly.data.get_path_of_test_ords(),
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
    import numpy as np

    for extraction in pickled_data_folder.glob("*"):
        df = pd.read_pickle(extraction)
        if df is None:
            continue

        # Columns: ['rxn_str_0', 'reactant_0', 'reactant_1', 'reactant_2', 'reactant_3', 'agent_0', 'agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5', 'solvent_0', 'solvent_1', 'solvent_2', 'temperature_0', 'rxn_time_0', 'product_0', 'yield_0'],
        # They're allowed to be strings or floats (depending on the col) or None
        for col in df.columns:
            series = df[col].replace({None: np.nan})
            if len(series.dropna()) == 0:
                continue
            elif "temperature" in col or "rxn_time" in col or "yield" in col:
                assert pd.api.types.is_float_dtype(series), f"failure for {col=}"
            else:
                assert pd.api.types.is_string_dtype(series), f"failure for {col=}"
