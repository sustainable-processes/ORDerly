import typing
import pytest


REPETITIONS = 5


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
        # [0,1, expected_labelled_reactants,expected_labelled_reagents,expected_labelled_solvents,expected_labelled_catalysts,expected_labelled_products_from_input,expected_non_smiles_names_list_additions],
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
    "file_name,rxn_idx,expected_rxn_str_reactants,expected_rxn_str_agents,expected_rxn_str_products,expected_rxn_str,expected_non_smiles_names_list_additions",
    (
        # ["ord_dataset-00005539a1e04c809a9a78647bea649c", 0, None],
        # TODO: allow the function to return None for this test case
        [
            "ord_dataset-0b70410902ae4139bd5d334881938f69",
            0,
            [
                "SCc1ccccc1",
                "O=[N+]([O-])c1ccc(Oc2ccc(C(F)(F)F)cc2Cl)cc1[N+](=O)[O-]",
            ],
            [
                "C1CCOC1",
                "[H-]",
                "[Na+]",
            ],
            ["O=[N+]([O-])c1ccc(Oc2ccc(C(F)(F)F)cc2Cl)cc1SCc1ccccc1"],
            "[CH2:1]([SH:8])[C:2]1[CH:7]=[CH:6][CH:5]=[CH:4][CH:3]=1.[H-].[Na+].[Cl:11][C:12]1[CH:30]=[C:29]([C:31]([F:34])([F:33])[F:32])[CH:28]=[CH:27][C:13]=1[O:14][C:15]1[CH:20]=[CH:19][C:18]([N+:21]([O-:23])=[O:22])=[C:17]([N+]([O-])=O)[CH:16]=1>O1CCCC1>[CH2:1]([S:8][C:17]1[CH:16]=[C:15]([O:14][C:13]2[CH:27]=[CH:28][C:29]([C:31]([F:34])([F:32])[F:33])=[CH:30][C:12]=2[Cl:11])[CH:20]=[CH:19][C:18]=1[N+:21]([O-:23])=[O:22])[C:2]1[CH:7]=[CH:6][CH:5]=[CH:4][CH:3]=1",
            [],
        ],
        [
            "ord_dataset-0bb2e99daa66408fb8dbd6a0781d241c",
            0,
            [
                "[O-]B([O-])[O-]",
                "[O-]B([O-])[O-]",
                "[O-]B([O-])[O-]",
                "[O-]B([O-])[O-]",
            ],
            [
                "O",
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
            ],
            [
                "[O-]B1OB2OB([O-])OB(O1)O2",
                "[Na+]",
                "[Na+]",
            ],
            "[B:1]([O-:4])([O-:3])[O-:2].[B:5]([O-:8])([O-:7])[O-:6].[B:9]([O-:12])([O-])[O-].[B:13]([O-])([O-])[O-].[Na+:17].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+]>O>[B:1]1([O-:4])[O:3][B:13]2[O:12][B:9]([O:6][B:5]([O-:8])[O:7]2)[O:2]1.[Na+:17].[Na+:17]",
            [],
        ],
    ),
)
@pytest.mark.parametrize("execution_number", range(REPETITIONS))
def test_extract_info_from_rxn(
    execution_number,
    file_name,
    rxn_idx,
    expected_rxn_str_reactants,
    expected_rxn_str_agents,
    expected_rxn_str_products,
    expected_rxn_str,
    expected_non_smiles_names_list_additions,
):
    rxn = get_rxn_func()(file_name=file_name, rxn_idx=rxn_idx)

    import orderly.extract.extractor

    (
        rxn_str_reactants,
        rxn_str_agents,
        rxn_str_products,
        rxn_str,
        non_smiles_names_list_additions,
    ) = orderly.extract.extractor.OrdExtractor.extract_info_from_rxn(rxn)
    # if rxn_info = None:
    #     assert expected_rxn_str == rxn_info, f"failure for {expected_rxn_str=} got {rxn_info}"
    # else:
    #     assert (expected_rxn_str_reactants == rxn_info)
    assert expected_rxn_str_reactants == rxn_str_reactants
    assert expected_rxn_str_agents == rxn_str_agents
    assert expected_rxn_str_products == rxn_str_products
    assert expected_rxn_str == rxn_str
    assert expected_non_smiles_names_list_additions == non_smiles_names_list_additions


@pytest.mark.parametrize(
    "file_name,rxn_idx,expected_temperature",
    (
        ["ord_dataset-00005539a1e04c809a9a78647bea649c", 0, 110],
        ["ord_dataset-0b70410902ae4139bd5d334881938f69", 0, None],
        ["ord_dataset-0bb2e99daa66408fb8dbd6a0781d241c", 0, 1100.0],
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


@pytest.mark.parametrize(
    "file_name,rxn_idx,expected_rxn_time",
    (
        ["ord_dataset-00005539a1e04c809a9a78647bea649c", 0, None],
        ["ord_dataset-0b70410902ae4139bd5d334881938f69", 0, None],
        ["ord_dataset-0bb2e99daa66408fb8dbd6a0781d241c", 0, 0.17],
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
    "file_name,rxn_idx,manual_replacements_dict,expected_reactants, expected_agents, expected_reagents,expected_solvents,expected_catalysts,expected_products,expected_yields,expected_temperature,expected_rxn_time,expected_rxn_str, expected_procedure_details, expected_names_list",
    (
        [
            "ord_dataset-00005539a1e04c809a9a78647bea649c",
            0,
            {},
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
        # [
        #     "ord_dataset-0bb2e99daa66408fb8dbd6a0781d241c", 0,
        #     {},
        #     ['[O-]B([O-])[O-].[O-]B([O-])[O-]'],
        #     ['[Na+]'],
        #     [],
        #     [],
        # ]
    ),
)
@pytest.mark.parametrize("execution_number", range(REPETITIONS))
def test_handle_reaction_object(
    execution_number,
    file_name,
    rxn_idx,
    manual_replacements_dict,
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
        # TODO: Update solvents_path to be not None

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
        rxn, manual_replacements_dict, solvents_set, metals
    )
    assert sorted(reactants) == sorted(
        expected_reactants
    ), f"failure for {sorted(expected_reactants)=} got {sorted(reactants)}"
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
        procedure_details == expected_procedure_details
    ), f"failure for {expected_procedure_details=} got {procedure_details}"


@pytest.mark.parametrize(
    "trust_labelling,use_multiprocessing,name_contains_substring,inverse_substring",
    (
        [False, True, "uspto", True],
        [False, True, "uspto", False],
        [True, False, "uspto", True],
        [True, True, None, True],
    ),
)
@pytest.mark.parametrize("execution_number", range(REPETITIONS))
def extraction_pipeline(
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
