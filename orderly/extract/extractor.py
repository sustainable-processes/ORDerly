import logging
import typing
import pathlib
import dataclasses

import numpy as np
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


def strip_filename(
    filename: str, replacements: typing.List[typing.Tuple[str, str]]
) -> str:
    for _from, _to in replacements:
        filename = filename.replace(_from, _to)
    return filename


@dataclasses.dataclass(kw_only=True)
class OrdExtractor:
    """
    Read in an ord file, check if it contains USPTO data, and then:
    1) Extract all the relevant data (raw): reactants, products, catalysts, reagents, yields, temp, time
    2) Canonicalise all the molecules
    3) Write to a pickle file
    """

    ord_file_path: pathlib.Path
    trust_labelling: bool
    manual_replacements_dict: typing.Dict[str, str]
    metals: typing.Optional[METALS] = None
    solvents_set: typing.Optional[typing.Set[SOLVENT]] = None
    filename: typing.Optional[str] = None
    contains_substring: typing.Optional[str] = None  # None or uspto
    inverse_contains_substring: bool = False

    def __post_init__(self):
        """loads in the data from the file and runs the extraction code to build the dataframe"""

        LOG.debug(f"Extracting data from {self.ord_file_path}")
        self.data = OrdExtractor.load_data(self.ord_file_path)

        if self.filename is None:
            self.filename = self.data.name

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
                self.filename = self.data.dataset_id

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

        if self.metals is None:
            self.metals = orderly.extract.defaults.get_metals_list()
        if self.solvents_set is None:
            self.solvents_set = orderly.extract.defaults.get_solvents_set()
        self.non_smiles_names_list = []
        self.full_df = self.build_full_df()
        LOG.debug(f"Got data from {self.ord_file_path}: {self.filename}")

    @staticmethod
    def load_data(
        ord_file_path: typing.Union[str, pathlib.Path]
    ) -> ord_dataset_pb2.Dataset:
        """
        Simply loads the ORD data.
        """
        if isinstance(ord_file_path, pathlib.Path):
            ord_file_path = str(ord_file_path)
        return ord_message_helpers.load_message(ord_file_path, ord_dataset_pb2.Dataset)

    @staticmethod
    def find_smiles(
        identifiers,
    ) -> typing.Tuple[
        typing.Optional[SMILES | MOLECULE_IDENTIFIER], typing.List[MOLECULE_IDENTIFIER]
    ]:
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
    ) -> typing.Tuple[typing.Optional[str], typing.Optional[bool]]:
        rxn_str_extended_smiles = None
        for rxn_ident in rxn.identifiers:
            if rxn_ident.type == 6:  # REACTION_CXSMILES
                rxn_str_extended_smiles = rxn_ident.value
                is_mapped = rxn_ident.is_mapped

        if rxn_str_extended_smiles is None:
            return None, None
        rxn_str = rxn_str_extended_smiles.split(" ")[
            0
        ]  # this is to get rid of the extended smiles info
        return rxn_str, is_mapped

    @staticmethod
    def extract_info_from_rxn(
        rxn: ord_reaction_pb2.Reaction,
    ) -> typing.Optional[
        typing.Tuple[REACTANTS, AGENTS, PRODUCTS, str, typing.List[MOLECULE_IDENTIFIER]]
    ]:
        """
        Input a reaction object, and return the reactants, agents, products, and the reaction smiles string
        """
        _ = rdkit_BlockLogs()
        rxn_str, is_mapped = OrdExtractor.get_rxn_string_and_is_mapped(rxn)
        if rxn_str is None:
            return None

        reactant_from_rxn, agent, product_from_rxn = rxn_str.split(">")
        

        reactant_from_rxn = reactant_from_rxn.split(".")
        agents = agent.split(".")
        product_from_rxn = product_from_rxn.split(".")
        

        non_smiles_names_list = []
        # We need molecules wihtout maping info, so we can compare them to the products
        reactant_from_rxn_without_mapping = []
        for smi in reactant_from_rxn:
            canon_smi = orderly.extract.canonicalise.get_canonicalised_smiles(
                smi, is_mapped
            )
            if canon_smi is None:
                canon_smi = smi
                non_smiles_names_list.append(smi)
            reactant_from_rxn_without_mapping.append(canon_smi)

        product_from_rxn_without_mapping = []
        for smi in product_from_rxn:
            canon_smi = orderly.extract.canonicalise.get_canonicalised_smiles(
                smi, is_mapped
            )
            if canon_smi is None:
                canon_smi = smi
                non_smiles_names_list.append(smi)
            product_from_rxn_without_mapping.append(canon_smi)

        cleaned_agents = []
        for smi in agents:
            canon_smi = orderly.extract.canonicalise.get_canonicalised_smiles(
                smi, is_mapped
            )
            if canon_smi is None:
                canon_smi = smi
                non_smiles_names_list.append(smi)
            cleaned_agents.append(canon_smi)
            


        reactants = []
        # Only the mapped reactants that also don't appear as products should be trusted as reactants
        # I.e. first check whether a reactant molecule has at least 1 mapped atom, and then check whether it appears in the products
        for r_map, r_clean in zip(reactant_from_rxn, reactant_from_rxn_without_mapping):
            # check reactant is mapped and also that it's not in the products
            mol = rdkit_Chem.MolFromSmiles(r_map)
            if mol != None:
                if any(
                    atom.HasProp("molAtomMapNumber") for atom in mol.GetAtoms()
                ) and (  # any(generator)
                    r_clean not in product_from_rxn_without_mapping
                ):
                    reactants.append(r_clean)
                else:
                    cleaned_agents.append(r_clean)
        products = [p for p in product_from_rxn_without_mapping if p not in reactants]
        products = [p for p in products if p not in cleaned_agents]
        
        return list(set(reactants)), list(set(cleaned_agents)), list(set(products)), rxn_str, non_smiles_names_list

    @staticmethod
    def rxn_input_extractor(
        rxn: ord_reaction_pb2.Reaction,
    ) -> typing.Tuple[
        REACTANTS,
        REAGENTS,
        SOLVENTS,
        CATALYSTS,
        PRODUCTS,
        typing.List[MOLECULE_IDENTIFIER],
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
            try:
                components = rxn.inputs[key].components
                for component in components:
                    rxn_role = component.reaction_role  # rxn role
                    identifiers = component.identifiers
                    smiles, non_smiles_names_list_additions = OrdExtractor.find_smiles(
                        identifiers
                    )
                    non_smiles_names_list += non_smiles_names_list_additions
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
            except IndexError:
                pass
        return (
            reactants,
            reagents,
            solvents,
            catalysts,
            products,
            non_smiles_names_list,
        )

    @staticmethod
    def rxn_outcomes_extractor(
        rxn: ord_reaction_pb2.Reaction,
    ) -> typing.Tuple[PRODUCTS, YIELDS, typing.List[MOLECULE_IDENTIFIER]]:
        """
        Extract reaction information from ORD output object (ord_reaction_pb2.Reaction.outcomes)
        """
        # products & yield
        yields = []
        products = []
        non_smiles_names_list = []

        products_obj = rxn.outcomes[0].products
        for product in products_obj:
            try:
                y = None
                identifiers = product.identifiers
                (
                    product_smiles,
                    non_smiles_names_list_additions,
                ) = OrdExtractor.find_smiles(identifiers)

                non_smiles_names_list += non_smiles_names_list_additions
                measurements = product.measurements
                for measurement in measurements:
                    if measurement.type == 3:  # YIELD
                        y = float(measurement.percentage.value)
                        y = round(y, 2)
                # people sometimes report a product such as '[Na+].[Na+].[O-]B1OB2OB([O-])OB(O1)O2' and then only report one yield, this is a problem...
                # We'll resolve this by moving the longest smiles string to the front of the list, then appending the yield to the front of the list, and padding with None to ensure that the lists are the same length

                # split the product string by dot and sort by descending length
                product_list = sorted(product_smiles.split("."), key=len, reverse=True)

                # create a list of the same length as product_list with y as the first value and None as the other values
                y_list = [y] + [None] * (len(product_list) - 1)

                products += product_list
                yields += y_list
            except IndexError:
                continue

        return products, yields, non_smiles_names_list

    @staticmethod
    def temperature_extractor(
        rxn: ord_reaction_pb2.Reaction,
    ) -> typing.Optional[TEMPERATURE_CELCIUS]:
        """
        Gets the temperature of a reaction in degrees celcius
        """
        try:
            # first look for the temperature as a number
            temp_unit = rxn.conditions.temperature.setpoint.units

            if temp_unit == 1:  # celcius
                return float(rxn.conditions.temperature.setpoint.value)

            elif temp_unit == 2:  # fahrenheit
                f = rxn.conditions.temperature.setpoint.value
                c = (f - 32) * 5 / 9
                return float(c)

            elif temp_unit == 3:  # kelvin
                k = rxn.conditions.temperature.setpoint.value
                c = k - 273.15
                return float(c)
            elif temp_unit == 0:  # unspecified
                # instead of using the setpoint, use the control type
                # temperatures are in celcius
                temp_control_type = rxn.conditions.temperature.control.type
                if temp_control_type == 2:  # AMBIENT
                    return 25.0
                elif temp_control_type == 6:  # ICE_BATH
                    return 0.0
                elif temp_control_type == 9:  # DRY_ICE_BATH
                    return -78.5
                elif temp_control_type == 11:  # LIQUID_NITROGEN
                    return -196.0
        except IndexError:
            pass
        return None  # No temperature found

    @staticmethod
    def rxn_time_extractor(rxn: ord_reaction_pb2.Reaction) -> typing.Optional[float]:
        try:
            if rxn.outcomes[0].reaction_time.units == 1:  # hour
                return round(float(rxn.outcomes[0].reaction_time.value), 2)
            elif rxn.outcomes[0].reaction_time.units == 3:  # seconds
                s = rxn.outcomes[0].reaction_time.value
                h = s / 3600
                return round(float(h), 2)
            elif rxn.outcomes[0].reaction_time.units == 2:  # minutes
                m = rxn.outcomes[0].reaction_time.value
                h = m / 60
                return round(float(h), 2)
            elif rxn.outcomes[0].reaction_time.units == 4:  # day
                d = rxn.outcomes[0].reaction_time.value
                h = d * 24
                return round(float(h), 2)
        except IndexError:
            pass
        return None  # no time found

    @staticmethod
    def apply_replacements_dict(smiles_list: list, manual_replacements_dict: dict):
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
        yields: YIELDS,
        use_labelling_if_extract_fails: bool = True,
    ) -> typing.Tuple[PRODUCTS, YIELDS]:
        """
        Resolve: yields are from rxn_outcomes(labelled_products), but we trust the products from the rxn_string
        """
        if len(rxn_str_products) != 0 and yields is not None:
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
        rxn_string_agents: AGENTS,
        catalysts: CATALYSTS,
        solvents: SOLVENTS,
        reagents: REAGENTS,
        metals: METALS,
        solvents_set: typing.Set[SOLVENT],
    ) -> typing.Tuple[AGENTS, SOLVENTS]:
        """
        Merge cat, solv, reag into agents list, and then extract solvents from agents list by cross-referencing to solvents_set. Then sort alphabetically and put metals (likely to be catalysts) first.
        """
        # merge the solvents, reagents, and catalysts into one list
        agents = []
        if rxn_string_agents is not None:
            agents += [a for a in rxn_string_agents if not None]
        if catalysts is not None:
            agents += [a for a in catalysts if not None]
        if solvents is not None:
            agents += [a for a in solvents if not None]
        if reagents is not None:
            agents += [a for a in reagents if not None]

        agents_set = set(agents)  # this includes the solvnts

        # build two new lists, one with the solvents, and one with the reagents+catalysts
        # Create a new set of solvents from agents_set
        solvents = agents_set.intersection(solvents_set)

        # Remove the solvents from agents_set
        agents = agents_set.difference(solvents)

        # I think we should add some ordering to the agents
        # What if we first order them alphabetically, and afterwards by putting the metals first in the list

        agents = sorted(list(agents))
        solvents = sorted(list(solvents))

        # Ideally we'd order the agents, so we have the catalysts (metal centre) first, then the ligands, then the bases and finally any reagents
        # We don't have a list of catalysts, and it's not straight forward to figure out if something is a catalyst or not (both chemically and computationally)
        # Instead, let's move all agents that contain a metal centre to the front of the list

        agents_with_metal = []
        agents_wo_metal = []
        for agent in agents:
            contains_metal = any(metal in agent for metal in metals)
            if contains_metal:
                agents_with_metal.append(agent)
            else:
                agents_wo_metal.append(agent)
        agents = agents_with_metal + agents_wo_metal
        return agents, solvents

    @staticmethod
    def handle_reaction_object(
        rxn: ord_reaction_pb2.Reaction,
        manual_replacements_dict: dict,
        solvents_set: typing.Set[SOLVENT],
        metals: METALS,
        trust_labelling: bool = False,
        use_labelling_if_extract_fails: bool = True,
    ) -> typing.Optional[
        typing.Tuple[
            REACTANTS,
            AGENTS,
            REAGENTS,
            SOLVENTS,
            CATALYSTS,
            PRODUCTS,
            YIELDS,
            TEMPERATURE_CELCIUS,
            RXN_TIME,
            RXN_STR,
            str,
            typing.List[MOLECULE_IDENTIFIER],
        ]
    ]:
        # handle rxn inputs: reactants, reagents etc

        # initilise empty
        reactants = []
        reagents = []
        solvents = []
        catalysts = []
        rxn_str_products = []
        labelled_products = []
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

        # TODO: labelled_products_from_input should either be empty or equal to labelled_products, if not, we should raise a warning

        # A reaction object has 3 data-sources: rxn_string, rxn_inputs, and rxn_outcomes; these sources should be in agreement, but it's still important to have a robust idea of how to resolve any disagreements
        rxn_str, is_mapped = OrdExtractor.get_rxn_string_and_is_mapped(rxn)

        if trust_labelling:
            reactants = labelled_reactants
            products = labelled_products
            yields = yields
            agents = []
            solvents = labelled_solvents
            reagents = labelled_reagents
            catalysts = labelled_catalysts
            is_mapped = False

        else:
            try:  # to extract info from the reaction string
                rxn_info = OrdExtractor.extract_info_from_rxn(rxn)
                if rxn_info is None:
                    raise ValueError("rxn_info is None")
                (
                    rxn_str_reactants,
                    rxn_str_agents,
                    rxn_str_products,
                    rxn_str,
                    rxn_non_smiles_names_list,
                ) = rxn_info
                reactants = list(set(rxn_str_reactants))
                # Resolve: yields are from rxn_outcomes, but we trust the products from the rxn_string
                rxn_str_products = list(set(rxn_str_products))
                products, yields = OrdExtractor.match_yield_with_product(
                    rxn_str_products, labelled_products, yields
                )
                
            except (ValueError, TypeError) as e:
                rxn_str_agents = []
                # we don't have a mapped reaction, so we have to just trust the labelled reactants, agents, and products
                # TODO: test what happens when value error is raised above
                if use_labelling_if_extract_fails:
                    reactants = labelled_reactants
                    products = labelled_products
                else:
                    return None

            # Merge conditions
            agents, solvents = OrdExtractor.merge_to_agents(
                rxn_str_agents,
                labelled_catalysts,
                labelled_solvents,
                labelled_reagents,
                metals,
                solvents_set,
            )
            reagents = []
            catalysts = []
        # extract temperature
        temperature = OrdExtractor.temperature_extractor(rxn)

        # extract rxn_time
        rxn_time = OrdExtractor.rxn_time_extractor(rxn)

        # clean the smiles

        def is_digit(x):
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
        ) -> typing.Tuple[
            REACTANTS | REAGENTS | SOLVENTS | CATALYSTS,
            typing.List[MOLECULE_IDENTIFIER],
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
        # TODO?: The canonicalisation is repeated! We extract information from rxn_str, and then apply logic to figure out what is a reactant/agent. So we canonicalise inside the extract_info_from_rxn function, but not within the input_extraction function, which is why we need to do it again here. This also means we add stuff to the non-smiles names list multiple times, so we need to do list(set()) on that list; all this is slightly inefficient, but shouldn't add that much overhead.
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

        # Apply the manual_replacements_dict to the agents, reagents, solvents, and catalysts
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

        procedure_details = rxn.notes.procedure_details
        rxn_non_smiles_names_list = list(set(rxn_non_smiles_names_list))

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
            rxn_non_smiles_names_list,
        )

    def build_rxn_lists(
        self,
    ) -> typing.Tuple[
        typing.List[RXN_STR],
        typing.List[REACTANTS],
        typing.List[AGENTS],
        typing.List[REAGENTS],
        typing.List[SOLVENTS],
        typing.List[CATALYSTS],
        typing.List[TEMPERATURE_CELCIUS],
        typing.List[RXN_TIME],
        typing.List[PRODUCTS],
        typing.List[YIELDS],
        typing.List[str],
    ]:
        rxn_str_all = []
        reactants_all = []
        products_all = []
        yields_all = []
        reagents_all = []
        agents_all = []
        solvents_all = []
        catalysts_all = []

        temperature_all = []
        rxn_time_all = []

        procedure_details_all = []
        rxn_non_smiles_names_list = []

        for rxn in self.data.reactions:
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
                rxn_non_smiles_names_list_additions,
            ) = OrdExtractor.handle_reaction_object(
                rxn,
                manual_replacements_dict=self.manual_replacements_dict,
                solvents_set=self.solvents_set,
                metals=self.metals,
                trust_labelling=self.trust_labelling,
            )
            rxn_non_smiles_names_list += rxn_non_smiles_names_list_additions

            if set(reactants) != set(
                products
            ):  # If the reactants and products are the same, then we don't want to add this reaction to our dataset
                rxn_str_all.append(rxn_str)
                reactants_all.append(reactants)
                products_all.append(products)
                yields_all.append(yields)
                temperature_all.append(temperature)
                rxn_time_all.append(rxn_time)
                procedure_details_all.append(procedure_details)
                agents_all.append(agents)
                solvents_all.append(solvents)
                reagents_all.append(reagents)
                catalysts_all.append(catalysts)

        return (
            rxn_str_all,
            reactants_all,
            agents_all,
            reagents_all,
            solvents_all,
            catalysts_all,
            temperature_all,
            rxn_time_all,
            products_all,
            yields_all,
            procedure_details_all,
        )

    def create_column_headers(
        self, df: pd.DataFrame, base_string: str
    ) -> typing.List[str]:
        """
        create the column headers for the df
        adds a base_string to the columns (prefix)
        """
        column_headers = []
        for i in range(len(df.columns)):
            column_headers.append(base_string + str(i))
        return column_headers

    def build_full_df(self) -> pd.DataFrame:
        headers = [
            "rxn_str_",
            "reactant_",
            "agent_",
            "reagent_",
            "solvent_",
            "catalyst_",
            "temperature_",
            "rxn_time_",
            "product_",
            "yield_",
        ]
        data_lists = self.build_rxn_lists()
        for i in range(len(headers)):
            new_df = pd.DataFrame(data_lists[i])
            df_headers = self.create_column_headers(new_df, headers[i])
            new_df.columns = df_headers
            if i == 0:
                full_df = new_df
            else:
                full_df = pd.concat([full_df, new_df], axis=1)
        return full_df
