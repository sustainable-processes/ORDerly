import logging
from typing import List, Dict, Tuple, Set, Optional, Union, Any
import pathlib
import dataclasses
import warnings

import pandas as pd

from ord_schema import message_helpers as ord_message_helpers
from ord_schema.proto import dataset_pb2 as ord_dataset_pb2
from ord_schema.proto import reaction_pb2 as ord_reaction_pb2

from rdkit import Chem as rdkit_Chem
from rdkit.rdBase import BlockLogs as rdkit_BlockLogs

import orderly.extract.defaults
import orderly.extract.canonicalise
from orderly.types import *

LOG = logging.getLogger(__name__)


def strip_filename(filename: str, replacements: List[Tuple[str, str]]) -> str:
    for _from, _to in replacements:
        filename = filename.replace(_from, _to)
    return filename


@dataclasses.dataclass(kw_only=True)
class OrdExtractor:
    """Read in an ord file, check if it contains data, and then:
    1) Extract all the relevant data (raw): reactants, products, catalysts, reagents, yields, temp, time
    2) Canonicalise all the molecules
    3) Write the dataframe to a parquet file
    """

    ord_file_path: pathlib.Path
    trust_labelling: bool
    manual_replacements_dict: MANUAL_REPLACEMENTS_DICT
    solvents_set: Optional[Set[SOLVENT]] = None
    filename: Optional[str] = None
    contains_substring: Optional[str] = None  # typically: None or uspto
    inverse_contains_substring: bool = False

    def __post_init__(self) -> None:
        """loads in the data from the file and runs the extraction code to build the dataframe"""

        LOG.debug(f"Extracting data from {self.ord_file_path}")
        self.data = OrdExtractor.load_data(self.ord_file_path)

        if self.filename is None:
            self.filename = str(self.data.name)

            self.filename = strip_filename(
                self.filename,
                replacements=[
                    ("/", "-fs-"),
                    (":", ""),
                    (" ", "_"),
                    (".", "-"),
                    ('"', ""),
                    ("'", ""),
                ],
            ).lower()

            if self.filename == "":
                LOG.debug(
                    f"No file name for dataset so using dataset_id={self.data.dataset_id}"
                )
                self.filename = str(self.data.dataset_id)

        # Get the date of the grant (proxy for when the data was collected)
        _grant_date = self.filename.split("uspto-grants-")
        grant_date: Optional[pd.Timestamp] = None
        if len(_grant_date) > 1:
            grant_date = pd.to_datetime(_grant_date[1], format="%Y_%M")

        self.non_smiles_names_list = self.full_df = None
        if self.contains_substring is not None:
            to_skip = self.contains_substring.lower() not in self.filename.lower()
            reason = "contains"
            if self.inverse_contains_substring:
                to_skip = not to_skip
                reason = "does not contain"
            reason += f" {self.contains_substring.lower()}"
            if to_skip:
                LOG.debug(
                    f"Skipping {self.ord_file_path}: {self.filename} as filename {reason}"
                )
                return

        if self.solvents_set is None:
            self.solvents_set = orderly.extract.defaults.get_solvents_set()
        self.full_df, self.non_smiles_names_list = self.build_full_df()

        self.full_df = self.full_df.assign(grant_date=grant_date)
        self.full_df.grant_date = pd.to_datetime(self.full_df.grant_date)

        LOG.debug(f"Got data from {self.ord_file_path}: {self.filename}")

    @staticmethod
    def load_data(ord_file_path: Union[str, pathlib.Path]) -> ord_dataset_pb2.Dataset:
        """
        Simply loads the ORD data.
        """
        if isinstance(ord_file_path, pathlib.Path):
            ord_file_path = str(ord_file_path)
        return ord_message_helpers.load_message(ord_file_path, ord_dataset_pb2.Dataset)

    @staticmethod
    def find_smiles(
        identifiers: REPEATEDCOMPOSITECONTAINER,
    ) -> Tuple[Optional[SMILES | MOLECULE_IDENTIFIER], List[MOLECULE_IDENTIFIER]]:
        """
        Search through the identifiers to return a smiles string, if this doesn't exist, search to return the English name, and if this doesn't exist, return None
        """

        non_smiles_names_list = (
            []
        )  # we dont require information on state of this object so can create a new one
        _ = rdkit_BlockLogs()
        for i in identifiers:
            if i.type == 2:
                canon_smi = orderly.extract.canonicalise.get_canonicalised_smiles(
                    i.value, is_mapped=True
                )
                if canon_smi is None:
                    canon_smi = i.value
                    non_smiles_names_list.append(i.value)
                return canon_smi, non_smiles_names_list
        for ii in identifiers:  # if there's no smiles, return the name
            if ii.type == 6:
                name = ii.value
                non_smiles_names_list.append(name)
                return name, non_smiles_names_list
        return None, non_smiles_names_list

    @staticmethod
    def get_rxn_string_and_is_mapped(
        rxn: ord_reaction_pb2.Reaction,
    ) -> Optional[Tuple[RXN_STR, bool]]:
        rxn_str_extended_smiles = None
        for rxn_ident in rxn.identifiers:
            if rxn_ident.type == 6:  # REACTION_CXSMILES
                rxn_str_extended_smiles = rxn_ident.value
                is_mapped = rxn_ident.is_mapped

        if rxn_str_extended_smiles is None:
            return None
        rxn_str = rxn_str_extended_smiles.split(" ")[
            0
        ]  # this is to get rid of the extended smiles info

        count = rxn_str.count(">")
        if count == 2:  # Finally, we need to check whether the reaction string is valid
            return RXN_STR(rxn_str), is_mapped
        else:
            return None

    @staticmethod
    def extract_info_from_rxn_str(
        rxn_str: RXN_STR, is_mapped: bool
    ) -> Tuple[REACTANTS, AGENTS, PRODUCTS, RXN_STR, List[MOLECULE_IDENTIFIER]]:
        """
        Input a reaction object, and return the reactants, agents, products, and the reaction smiles string
        """
        _ = rdkit_BlockLogs()

        reactant_from_rxn, agent, product_from_rxn = rxn_str.split(">")

        reactants_from_rxn = reactant_from_rxn.split(".")
        agents = agent.split(".")
        products_from_rxn = product_from_rxn.split(".")
        del reactant_from_rxn
        del agent
        del product_from_rxn

        non_smiles_names_list: List[MOLECULE_IDENTIFIER] = []
        # We need molecules wihtout maping info, so we can compare them to the products
        reactants_from_rxn_without_mapping: CANON_REACTANTS = []
        for smi in reactants_from_rxn:
            canon_smi = orderly.extract.canonicalise.get_canonicalised_smiles(
                smi, is_mapped
            )
            if canon_smi is None:
                canon_smi = smi
                non_smiles_names_list.append(smi)
            reactants_from_rxn_without_mapping.append(canon_smi)
        # assert len(reactants_from_rxn) == len(reactants_from_rxn_without_mapping)

        products_from_rxn_without_mapping: CANON_PRODUCTS = []
        for smi in products_from_rxn:
            canon_smi = orderly.extract.canonicalise.get_canonicalised_smiles(
                smi, is_mapped
            )
            if canon_smi is None:
                canon_smi = smi
                non_smiles_names_list.append(smi)
            products_from_rxn_without_mapping.append(canon_smi)
        # assert len(products_from_rxn) == len(products_from_rxn_without_mapping)

        cleaned_agents: CANON_AGENTS = []
        for smi in agents:
            canon_smi = orderly.extract.canonicalise.get_canonicalised_smiles(
                smi, is_mapped
            )
            if canon_smi is None:
                canon_smi = smi
                non_smiles_names_list.append(smi)
            cleaned_agents.append(canon_smi)
        # assert len(agents) == len(cleaned_agents)

        reactants = []
        # Only the mapped reactants that also don't appear as products should be trusted as reactants
        # I.e. first check whether a reactant molecule has at least 1 mapped atom, and then check whether it appears in the products
        for r_map, r_clean in zip(
            reactants_from_rxn, reactants_from_rxn_without_mapping
        ):
            # check reactant is mapped and also that it's not in the products
            mol = rdkit_Chem.MolFromSmiles(r_map)
            if mol != None:
                if any(
                    atom.HasProp("molAtomMapNumber") for atom in mol.GetAtoms()
                ) and (  # any(generator)
                    r_clean not in products_from_rxn_without_mapping
                ):
                    reactants.append(r_clean)
                else:
                    cleaned_agents.append(r_clean)
        products = [p for p in products_from_rxn_without_mapping if p not in reactants]
        products = [p for p in products if p not in cleaned_agents]

        return (
            sorted(list(set(reactants))),
            sorted(list(set(cleaned_agents))),
            sorted(list(set(products))),
            rxn_str,
            non_smiles_names_list,
        )

    @staticmethod
    def rxn_input_extractor(
        rxn: ord_reaction_pb2.Reaction,
    ) -> Tuple[
        REACTANTS,
        REAGENTS,
        SOLVENTS,
        CATALYSTS,
        PRODUCTS,
        List[MOLECULE_IDENTIFIER],
    ]:
        """
        Extract reaction information from ORD input object (ord_reaction_pb2.Reaction.inputs)
        """
        # loop through the keys (components) in the 'dict' style data struct. One key may contain multiple components/molecules.
        non_smiles_names_list = (
            []
        )  # we dont require information on state of this object so can create a new one
        # initialise lists
        reactants = []
        reagents = []
        solvents = []
        catalysts = []
        products = []

        for key in rxn.inputs:
            components = rxn.inputs[key].components

            for component in components:
                rxn_role = component.reaction_role  # rxn role
                identifiers = component.identifiers

                smiles, non_smiles_names_list_additions = OrdExtractor.find_smiles(
                    identifiers
                )
                non_smiles_names_list += non_smiles_names_list_additions
                if smiles is None:
                    LOG.debug(f"No smiles or english name found for {identifiers=}")
                    continue
                if rxn_role == 1:  # NB: Reagents may be misclassified as reactants
                    reactants += [r for r in smiles.split(".")]
                elif rxn_role == 2:  # reagent
                    reagents += [r for r in smiles.split(".")]
                elif rxn_role == 3:  # solvent
                    solvents += [r for r in smiles.split(".")]
                elif rxn_role == 4:  # catalyst
                    catalysts += [r for r in smiles.split(".")]
                elif rxn_role in [5, 6, 7]:
                    # 5=workup, 6=internal standard, 7=authentic standard. don't care about these
                    continue
                elif rxn_role == 8:  # product
                    # there are typically no products recorded in rxn_role == 8, they're all stored in "outcomes"
                    products += [r for r in smiles.split(".")]

        return (
            sorted(reactants),
            sorted(reagents),
            sorted(solvents),
            sorted(catalysts),
            sorted(products),
            non_smiles_names_list,
        )

    @staticmethod
    def rxn_outcomes_extractor(
        rxn: ord_reaction_pb2.Reaction,
    ) -> Tuple[PRODUCTS, YIELDS, List[MOLECULE_IDENTIFIER]]:
        """
        Extract reaction information from ORD output object (ord_reaction_pb2.Reaction.outcomes)
        """
        # products & yield
        yields: YIELDS = []
        products = []
        non_smiles_names_list = []

        products_obj = rxn.outcomes[0].products
        for product in products_obj:
            y = None
            identifiers = product.identifiers
            (
                product_smiles,
                non_smiles_names_list_additions,
            ) = OrdExtractor.find_smiles(identifiers)

            if product_smiles is None:
                continue

            non_smiles_names_list += non_smiles_names_list_additions
            measurements = product.measurements
            for measurement in measurements:
                if measurement.type == 3:  # YIELD
                    y = float(measurement.percentage.value)
                    y = YIELD(round(y, 2))
                    continue
            # people sometimes report a product such as '[Na+].[Na+].[O-]B1OB2OB([O-])OB(O1)O2' and then only report one yield, this is a problem...
            # We'll resolve this by moving the longest smiles string to the front of the list, then appending the yield to the front of the list, and padding with None to ensure that the lists are the same length

            # split the product string by dot and sort by descending length
            product_list = sorted(product_smiles.split("."), key=len, reverse=True)

            # create a list of the same length as product_list with y as the first value and None as the other values
            y_list = [y] + [None] * (len(product_list) - 1)

            products += product_list
            yields += y_list  # type: ignore

        return products, yields, non_smiles_names_list

    @staticmethod
    def temperature_extractor(
        rxn: ord_reaction_pb2.Reaction,
    ) -> Optional[TEMPERATURE_CELCIUS]:
        """
        Gets the temperature of a reaction in degrees celcius
        """
        # first look for the temperature as a number
        temp_unit = rxn.conditions.temperature.setpoint.units

        if temp_unit == 1:  # celcius
            return TEMPERATURE_CELCIUS(float(rxn.conditions.temperature.setpoint.value))
        elif temp_unit == 2:  # fahrenheit
            f = rxn.conditions.temperature.setpoint.value
            c = (f - 32) * 5 / 9
            return TEMPERATURE_CELCIUS(float(c))
        elif temp_unit == 3:  # kelvin
            k = rxn.conditions.temperature.setpoint.value
            c = k - 273.15
            return TEMPERATURE_CELCIUS(float(c))
        elif temp_unit == 0:  # unspecified
            # instead of using the setpoint, use the control type
            # temperatures are in celcius
            temp_control_type = rxn.conditions.temperature.control.type
            if temp_control_type == 2:  # AMBIENT
                return TEMPERATURE_CELCIUS(25.0)
            elif temp_control_type == 6:  # ICE_BATH
                return TEMPERATURE_CELCIUS(0.0)
            elif temp_control_type == 9:  # DRY_ICE_BATH
                return TEMPERATURE_CELCIUS(-78.5)
            elif temp_control_type == 11:  # LIQUID_NITROGEN
                return TEMPERATURE_CELCIUS(-196.0)
        return None  # No temperature found

    @staticmethod
    def rxn_time_extractor(rxn: ord_reaction_pb2.Reaction) -> Optional[RXN_TIME]:
        if rxn.outcomes[0].reaction_time.units == 1:  # hour
            return RXN_TIME(round(float(rxn.outcomes[0].reaction_time.value), 2))
        elif rxn.outcomes[0].reaction_time.units == 2:  # minutes
            m = rxn.outcomes[0].reaction_time.value
            h = m / 60
            return RXN_TIME(round(float(h), 2))
        elif rxn.outcomes[0].reaction_time.units == 3:  # seconds
            s = rxn.outcomes[0].reaction_time.value
            h = s / 3600
            return RXN_TIME(round(float(h), 2))
        elif rxn.outcomes[0].reaction_time.units == 4:  # day
            d = rxn.outcomes[0].reaction_time.value
            h = d * 24
            return RXN_TIME(round(float(h), 2))
        else:
            return None  # no time found

    @staticmethod
    def procedure_details_extractor(
        rxn: ord_reaction_pb2.Reaction,
    ) -> str:  # TODO check does it return empty string or none
        procedure_details = rxn.notes.procedure_details
        return str(procedure_details)

    @staticmethod
    def date_of_experiment_extractor(
        rxn: ord_reaction_pb2.Reaction,
    ) -> Optional[pd.Timestamp]:
        _date_of_experiment = rxn.provenance.experiment_start.value
        if len(_date_of_experiment) == 0:
            date_of_experiment = None
        else:  # we trust that it is a string that is convertible to a pd.Timestamp
            date_of_experiment = pd.to_datetime(_date_of_experiment, format="%m/%d/%Y")
        return date_of_experiment

    @staticmethod
    def apply_replacements_dict(
        smiles_list: List[MOLECULE_IDENTIFIER],
        manual_replacements_dict: MANUAL_REPLACEMENTS_DICT,
    ) -> List[SMILES]:
        smiles_list = [
            x
            for x in pd.Series(smiles_list, dtype=pd.StringDtype())
            .map(
                lambda x: manual_replacements_dict.get(x, x),
                na_action="ignore",
            )
            .tolist()
            if x is not None
        ]
        return smiles_list

    @staticmethod
    def match_yield_with_product(
        rxn_str_products: PRODUCTS,
        labelled_products: PRODUCTS,
        yields: Optional[YIELDS],
        use_labelling_if_extract_fails: bool = True,
    ) -> Tuple[PRODUCTS, Optional[YIELDS]]:
        """
        Resolve: yields are from rxn_outcomes(labelled_products), but we trust the products from the rxn_string
        """
        if (len(rxn_str_products) != 0) and (yields is not None):
            reordered_yields = []
            for rxn_str_prod in rxn_str_products:
                added = False
                for ii, lab_prod in enumerate(labelled_products):
                    if rxn_str_prod == lab_prod:
                        reordered_yields.append(yields[ii])
                        added = True
                        break
                if not added:
                    reordered_yields.append(None)
            return rxn_str_products, reordered_yields
        elif use_labelling_if_extract_fails:
            return labelled_products, yields
        else:
            return [], []

    @staticmethod
    def merge_to_agents(
        rxn_string_agents: Optional[AGENTS],
        catalysts: Optional[CATALYSTS],
        solvents: Optional[SOLVENTS],
        reagents: Optional[REAGENTS],
        solvents_set: Set[SOLVENT],
    ) -> Tuple[AGENTS, SOLVENTS]:
        """
        Merge cat, solv, reag into agents list, and then extract solvents from agents list by cross-referencing to solvents_set. Then sort alphabetically and put transition metals (likely to be catalysts) first.
        """
        # merge the solvents, reagents, and catalysts into one list
        agents = []
        if rxn_string_agents is not None:
            agents += [a for a in rxn_string_agents if a is not None]
        if catalysts is not None:
            agents += [a for a in catalysts if a is not None]
        if solvents is not None:
            agents += [a for a in solvents if a is not None]
        if reagents is not None:
            agents += [a for a in reagents if a is not None]

        agents_set = set(agents)  # this includes the solvnts

        # build two new lists, one with the solvents, and one with the reagents+catalysts
        # Create a new set of solvents from agents_set
        _solvents = agents_set.intersection(solvents_set)

        # Remove the solvents from agents_set
        _agents = agents_set.difference(_solvents)

        # I think we should add some ordering to the agents
        # What if we first order them alphabetically, and afterwards by putting the transition metals first in the list

        agents = sorted(list(_agents))
        solvents = sorted(list(_solvents))
        del _agents, _solvents  # for mypy

        # Ideally we'd order the agents, so we have the catalysts (metal centre) first, then the ligands, then the bases and finally any reagents
        # We don't have a list of catalysts, and it's not straight forward to figure out if something is a catalyst or not (both chemically and computationally)
        # Instead, let's move all agents that contain a transition metal centre to the front of the list

        agents_with_transition_metal = []
        agents_wo_transition_metal = []
        for agent in agents:
            agent_has_transition_metal = orderly.extract.defaults.has_transition_metal(
                agent
            )
            if agent_has_transition_metal:
                agents_with_transition_metal.append(agent)
            else:
                agents_wo_transition_metal.append(agent)
        agents = agents_with_transition_metal + agents_wo_transition_metal
        return agents, solvents

    @staticmethod
    def handle_reaction_object(
        rxn: ord_reaction_pb2.Reaction,
        manual_replacements_dict: MANUAL_REPLACEMENTS_DICT,
        solvents_set: Set[SOLVENT],
        trust_labelling: bool = False,
        use_labelling_if_extract_fails: bool = True,
        include_unadded_labelled_agents: bool = True,
    ) -> Optional[
        Tuple[
            REACTANTS,
            AGENTS,
            REAGENTS,
            SOLVENTS,
            CATALYSTS,
            PRODUCTS,
            YIELDS,
            Optional[TEMPERATURE_CELCIUS],
            Optional[RXN_TIME],
            Optional[RXN_STR],
            str,
            Optional[pd.Timestamp],
            List[MOLECULE_IDENTIFIER],
        ]
    ]:
        """
        An ORD rxn object has 3 sources of rxn data: the rxn string, rxn.inputs, and rxn.outcomes.
        If trust_labelling is True, we trust the labelling of the rxn.inputs and rxn.outcomes, and don't use the rxn string.
        If trust_labelling is False (default), we determine reactants, agents, solvents, and products, from the rxn string by looking at the mapping of the reaction (hence why we trust the rxn string more than the inputs/outcomes labelling, and this behaviour is set to default). However, the rxn.inputs and rxn.outcomes may contain info not contained in the rxn string:
            - If use_labelling_if_extract_fails is True, we use the labelling of the rxn.inputs and rxn.outcomes instead of simply returning None
            - If include_unadded_labelled_agents is True, we look through the rxn.inputs for any agents that were not added to the reactants, agents, solvents, or products, and add them to the agents list
        """
        # handle rxn inputs: reactants, reagents etc

        # initilise empty
        reactants: REACTANTS = []
        reagents: REAGENTS = []
        solvents: SOLVENTS = []
        catalysts: CATALYSTS = []
        rxn_str_products: PRODUCTS = []
        labelled_products: PRODUCTS = []
        rxn_non_smiles_names_list = []

        (
            labelled_reactants,
            labelled_reagents,
            labelled_solvents,
            labelled_catalysts,
            labelled_products_from_input,  # I'm not sure what to do with this, it doesn't make sense for people to have put a product as an input, so this list should be empty anyway
            non_smiles_names_list_additions,
        ) = OrdExtractor.rxn_input_extractor(rxn)
        rxn_non_smiles_names_list += non_smiles_names_list_additions

        (
            labelled_products,
            yields,
            non_smiles_names_list_additions,
        ) = OrdExtractor.rxn_outcomes_extractor(rxn)
        rxn_non_smiles_names_list += non_smiles_names_list_additions

        if (labelled_products_from_input != []) and (
            labelled_products_from_input != labelled_products
        ):  # we would expect the labelled products from input to be empty, but if it's not, we should check that it's the same as the labelled products
            if len(labelled_products_from_input) != len(labelled_products):
                warnings.warn(
                    "The number of products in rxn.inputs and rxn.outcomes do not match"
                )
            for idx, (mole_id_from_input, mole_id_from_outcomes) in enumerate(
                zip(sorted(labelled_products_from_input), sorted(labelled_products))
            ):
                smi_from_input = orderly.extract.canonicalise.get_canonicalised_smiles(
                    mole_id_from_input, is_mapped=False
                )
                smi_from_outcomes = (
                    orderly.extract.canonicalise.get_canonicalised_smiles(
                        mole_id_from_outcomes, is_mapped=False
                    )
                )
                if smi_from_input != smi_from_outcomes:
                    warnings.warn(
                        f"The smiles do not match for {idx=}: {smi_from_input=}!={smi_from_outcomes=}"
                    )

        # A reaction object has 3 data-sources: rxn_string, rxn_inputs, and rxn_outcomes; these sources should be in agreement, but it's still important to have a robust idea of how to resolve any disagreements
        _rxn_str = OrdExtractor.get_rxn_string_and_is_mapped(rxn)
        if _rxn_str is None:
            rxn_str, is_mapped = None, False
        else:
            rxn_str, is_mapped = _rxn_str

        # Get all the molecules
        if trust_labelling or (rxn_str is None and use_labelling_if_extract_fails):
            reactants = labelled_reactants
            products = labelled_products
            yields = yields
            agents: AGENTS = []
            solvents = labelled_solvents
            reagents = labelled_reagents
            catalysts = labelled_catalysts
            is_mapped = False

        elif rxn_str is not None:
            # extract info from the reaction string
            rxn_info = OrdExtractor.extract_info_from_rxn_str(rxn_str, is_mapped)
            (
                reactants,
                agents,
                _products,
                rxn_str,
                rxn_non_smiles_names_list,
            ) = rxn_info
            # Resolve: yields are from rxn_outcomes, but we trust the products from the rxn_string
            products, _yields = OrdExtractor.match_yield_with_product(
                _products, labelled_products, yields
            )
            if _yields is None:
                _yields = []
            yields = _yields

            if (
                include_unadded_labelled_agents
            ):  # Add any agents that were not added to the reactants, agents, or solvents
                # merge all the lists
                all_labelled_molecules = (
                    labelled_reactants
                    + labelled_products
                    + labelled_solvents
                    + labelled_reagents
                    + labelled_catalysts
                )
                # remove duplicates
                all_labelled_molecules = sorted(list(set(all_labelled_molecules)))
                # remove any molecules that are already in the reactants, agents, or solvents
                molecules_unique_to_labelled_data = [
                    x
                    for x in all_labelled_molecules
                    if x not in reactants + agents + solvents + products
                ]
                agents += molecules_unique_to_labelled_data

        else:
            return None

        if (
            not trust_labelling
        ):  # if we don't trust the labelling, we should merge the labelled data with the extracted data into just 'agents' and 'solvents'
            # Merge conditions
            agents, solvents = OrdExtractor.merge_to_agents(
                agents,
                labelled_catalysts,
                labelled_solvents,
                labelled_reagents,
                solvents_set,
            )
            reagents = []
            catalysts = []

        # clean the smiles

        def is_digit(x: Optional[str]) -> Optional[bool]:
            if x is None:
                return None
            elif isinstance(x, str):
                return x.isdigit()
            else:
                raise ValueError(f"Expected a string or None, got {type(x)}")

        # remove molecules that are integers
        reactants = [x for x in reactants if not is_digit(x)]
        agents = [x for x in agents if not is_digit(x)]
        reagents = [x for x in reagents if not is_digit(x)]
        solvents = [x for x in solvents if not is_digit(x)]
        catalysts = [x for x in catalysts if not is_digit(x)]

        def canonicalise_and_get_non_smiles_names(
            mole_id_list: REACTANTS | REAGENTS | SOLVENTS | CATALYSTS,
            is_mapped: bool = False,
        ) -> Tuple[
            REACTANTS | REAGENTS | SOLVENTS | CATALYSTS,
            List[MOLECULE_IDENTIFIER],
        ]:
            """Canonicalise the smiles and return the identifier (either SMILES or non-SMILES) as well as a list of non-SMILES names"""
            assert isinstance(mole_id_list, list)
            non_smiles_names_list_additions = []
            for idx, mole_id in enumerate(mole_id_list):
                smi = orderly.extract.canonicalise.get_canonicalised_smiles(
                    mole_id, is_mapped=is_mapped
                )
                if smi is None:
                    non_smiles_names_list_additions.append(mole_id)
                    mole_id_list[idx] = mole_id
                else:
                    mole_id_list[idx] = smi
            return mole_id_list, non_smiles_names_list_additions

        # Reactants and products might be mapped, but agents are not
        # TODO?: The canonicalisation is repeated! We extract information from rxn_str, and then apply logic to figure out what is a reactant/agent. So we canonicalise inside the extract_info_from_rxn_str function, but not within the input_extraction function, which is why we need to do it again here. This also means we add stuff to the non-smiles names list multiple times, so we need to do list(set()) on that list; all this is slightly inefficient, but shouldn't add that much overhead.
        (
            reactants,
            non_smiles_names_list_additions,
        ) = canonicalise_and_get_non_smiles_names(
            mole_id_list=reactants, is_mapped=is_mapped
        )
        rxn_non_smiles_names_list += non_smiles_names_list_additions

        (
            agents,
            non_smiles_names_list_additions,
        ) = canonicalise_and_get_non_smiles_names(
            mole_id_list=agents, is_mapped=is_mapped
        )
        rxn_non_smiles_names_list += non_smiles_names_list_additions

        (
            reagents,
            non_smiles_names_list_additions,
        ) = canonicalise_and_get_non_smiles_names(
            mole_id_list=reagents, is_mapped=is_mapped
        )
        rxn_non_smiles_names_list += non_smiles_names_list_additions

        (
            solvents,
            non_smiles_names_list_additions,
        ) = canonicalise_and_get_non_smiles_names(
            mole_id_list=solvents, is_mapped=is_mapped
        )
        rxn_non_smiles_names_list += non_smiles_names_list_additions

        (
            catalysts,
            non_smiles_names_list_additions,
        ) = canonicalise_and_get_non_smiles_names(
            mole_id_list=catalysts, is_mapped=is_mapped
        )
        rxn_non_smiles_names_list += non_smiles_names_list_additions

        (
            products,
            non_smiles_names_list_additions,
        ) = canonicalise_and_get_non_smiles_names(
            mole_id_list=products, is_mapped=is_mapped
        )
        rxn_non_smiles_names_list += non_smiles_names_list_additions

        # Apply the manual_replacements_dict to the reactants, agents, reagents, solvents, and catalysts
        reactants = OrdExtractor.apply_replacements_dict(
            reactants, manual_replacements_dict=manual_replacements_dict
        )
        agents = OrdExtractor.apply_replacements_dict(
            agents, manual_replacements_dict=manual_replacements_dict
        )
        reagents = OrdExtractor.apply_replacements_dict(
            reagents, manual_replacements_dict=manual_replacements_dict
        )
        solvents = OrdExtractor.apply_replacements_dict(
            solvents, manual_replacements_dict=manual_replacements_dict
        )
        catalysts = OrdExtractor.apply_replacements_dict(
            catalysts, manual_replacements_dict=manual_replacements_dict
        )

        # if reagent appears in reactant list, remove it
        # Since we're technically not sure whether something is a reactant (contributes atoms) or a reagent/solvent/catalyst (does not contribute atoms), it's probably more cautious to remove molecules that appear in both lists from the reagents/solvents/catalysts list rather than the reactants list

        agents = [a for a in agents if a not in reactants]
        reagents = [r for r in reagents if r not in reactants]
        solvents = [s for s in solvents if s not in reactants]
        catalysts = [c for c in catalysts if c not in reactants]

        procedure_details = OrdExtractor.procedure_details_extractor(rxn)
        date_of_experiment = OrdExtractor.date_of_experiment_extractor(rxn)
        temperature = OrdExtractor.temperature_extractor(rxn)
        rxn_time = OrdExtractor.rxn_time_extractor(rxn)

        rxn_non_smiles_names_list = sorted(list(set(rxn_non_smiles_names_list)))

        return (
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
            date_of_experiment,
            rxn_non_smiles_names_list,
        )

    def build_rxn_lists(
        self,
    ) -> Tuple[
        Dict[
            str,
            Union[
                List[Optional[RXN_STR]],
                List[REACTANTS],
                List[AGENTS],
                List[REAGENTS],
                List[SOLVENTS],
                List[CATALYSTS],
                List[Optional[TEMPERATURE_CELCIUS]],
                List[Optional[RXN_TIME]],
                List[PRODUCTS],
                List[YIELDS],
                List[str],
                List[Optional[pd.Timestamp]],
            ],
        ],
        List[MOLECULE_IDENTIFIER],
    ]:
        rxn_non_smiles_names_list: List[MOLECULE_IDENTIFIER] = []

        # mypy struggles with the dict so we just ignore here
        rxn_lists = {  # type: ignore
            "rxn_str": [],
            "reactant": [],
            "agent": [],
            "reagent": [],
            "solvent": [],
            "catalyst": [],
            "temperature": [],
            "rxn_time": [],
            "product": [],
            "yield": [],
            "procedure_details": [],
            "date_of_experiment": [],
        }

        assert self.solvents_set is not None

        for rxn in self.data.reactions:
            extracted_reaction = OrdExtractor.handle_reaction_object(
                rxn,
                manual_replacements_dict=self.manual_replacements_dict,
                solvents_set=self.solvents_set,
                trust_labelling=self.trust_labelling,
            )
            if extracted_reaction is None:
                continue
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
                date_of_experiment,
                rxn_non_smiles_names_list_additions,
            ) = extracted_reaction

            rxn_non_smiles_names_list += rxn_non_smiles_names_list_additions

            if set(reactants) != set(
                products
            ):  # If the reactants and products are the same, then we don't want to add this reaction to our dataset
                rxn_lists["rxn_str"].append(rxn_str)
                rxn_lists["reactant"].append(reactants)
                rxn_lists["agent"].append(agents)
                rxn_lists["reagent"].append(reagents)
                rxn_lists["solvent"].append(solvents)
                rxn_lists["catalyst"].append(catalysts)
                rxn_lists["temperature"].append(temperature)
                rxn_lists["rxn_time"].append(rxn_time)
                rxn_lists["product"].append(products)
                rxn_lists["yield"].append(yields)
                rxn_lists["procedure_details"].append(procedure_details)
                rxn_lists["date_of_experiment"].append(date_of_experiment)

        return rxn_lists, rxn_non_smiles_names_list

    @staticmethod
    def _create_column_headers(num_cols: int, base_string: str) -> List[str]:
        """
        create the column headers for the df
        adds a base_string to the columns (prefix)
        """
        return [f"{base_string}_{i}" for i in range(num_cols)]

    @staticmethod
    def _to_dataframe(cols: List[Any], base_string: str | List[str]) -> pd.DataFrame:
        df = pd.DataFrame(cols)
        if isinstance(base_string, str):
            df.columns = OrdExtractor._create_column_headers(
                num_cols=df.shape[1], base_string=base_string
            )
        else:
            assert df.shape[1] == 1
            df.columns = base_string
        return df

    def build_full_df(
        self,
    ) -> Tuple[pd.DataFrame, List[MOLECULE_IDENTIFIER]]:
        data_lists, rxn_non_smiles_names_list = self.build_rxn_lists()
        LOG.info("Build rxn lists")

        dfs = []
        dfs.append(
            OrdExtractor._to_dataframe(data_lists["rxn_str"], base_string=["rxn_str"])
            .astype("string")
            .astype(object)
        )
        dfs.append(
            OrdExtractor._to_dataframe(data_lists["reactant"], base_string="reactant")
            .astype("string")
            .astype(object)
        )
        dfs.append(
            OrdExtractor._to_dataframe(data_lists["agent"], base_string="agent")
            .astype("string")
            .astype(object)
        )
        dfs.append(
            OrdExtractor._to_dataframe(data_lists["reagent"], base_string="reagent")
            .astype("string")
            .astype(object)
        )
        dfs.append(
            OrdExtractor._to_dataframe(data_lists["solvent"], base_string="solvent")
            .astype("string")
            .astype(object)
        )
        dfs.append(
            OrdExtractor._to_dataframe(data_lists["catalyst"], base_string="catalyst")
            .astype("string")
            .astype(object)
        )
        dfs.append(
            OrdExtractor._to_dataframe(
                data_lists["temperature"], base_string=["temperature"]
            ).astype("float")
        )
        dfs.append(
            OrdExtractor._to_dataframe(
                data_lists["rxn_time"], base_string=["rxn_time"]
            ).astype("float")
        )  # TODO do we extract multiple rxn times?
        dfs.append(
            OrdExtractor._to_dataframe(data_lists["product"], base_string="product")
            .astype("string")
            .astype(object)
        )
        dfs.append(
            OrdExtractor._to_dataframe(data_lists["yield"], base_string="yield").astype(
                "float"
            )
        )
        dfs.append(
            OrdExtractor._to_dataframe(
                data_lists["procedure_details"], base_string=["procedure_details"]
            )
            .astype("string")
            .astype(object)
        )
        dfs.append(
            OrdExtractor._to_dataframe(
                data_lists["date_of_experiment"], base_string=["date_of_experiment"]
            ).apply(pd.to_datetime, errors="coerce")
        )
        LOG.info("Constructed dict of dfs")

        full_df = pd.concat(dfs, axis=1)
        LOG.info("Constructed df")
        return full_df, rxn_non_smiles_names_list
