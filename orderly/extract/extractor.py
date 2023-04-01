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
    merge_cat_solv_reag: bool
    manual_replacements_dict: typing.Dict[str, str]
    metals: typing.Optional[METALS] = None
    solvents_set: typing.Optional[typing.Set[SOLVENT]] = None
    filename: typing.Optional[str] = None
    contains_substring: typing.Optional[str] = None  # None or uspto
    inverse_contains_substring: bool = False

    def __post_init__(self):
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
        if isinstance(ord_file_path, pathlib.Path):
            ord_file_path = str(ord_file_path)
        return ord_message_helpers.load_message(ord_file_path, ord_dataset_pb2.Dataset)

    @staticmethod
    def find_smiles(
        identifiers,
    ) -> typing.Tuple[
        typing.Optional[SMILES | MOLECULE_IDENTIFIER], typing.List[MOLECULE_IDENTIFIER]
    ]:
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
    def extract_info_from_rxn_str(
        rxn: ord_reaction_pb2.Reaction,
    ) -> typing.Tuple[
        REACTANTS, REAGENTS, PRODUCTS, str, typing.List[MOLECULE_IDENTIFIER]
    ]:
        _ = rdkit_BlockLogs()
        mapped_rxn_extended_smiles = rxn.identifiers[0].value
        mapped_rxn = mapped_rxn_extended_smiles.split(" ")[
            0
        ]  # this is to get rid of the extended smiles info

        reactant_from_rxn, reagent, mapped_product = mapped_rxn.split(">")

        reactant_from_rxn = [r for r in reactant_from_rxn.split(".")]
        reagents = [r for r in reagent.split(".")]
        mapped_products = [p for p in mapped_product.split(".")]

        non_smiles_names_list = []
        # We need molecules wihtout maping info, so we can compare them to the products
        reactant_from_rxn_without_mapping = []
        for smi in reactant_from_rxn:
            canon_smi = orderly.extract.canonicalise.get_canonicalised_smiles(
                smi, is_mapped=True
            )
            if canon_smi is None:
                canon_smi = smi
                non_smiles_names_list.append(smi)
            reactant_from_rxn_without_mapping.append(canon_smi)

        product_from_rxn_without_mapping = []
        for smi in mapped_products:
            canon_smi = orderly.extract.canonicalise.get_canonicalised_smiles(
                smi, is_mapped=True
            )
            if canon_smi is None:
                canon_smi = smi
                non_smiles_names_list.append(smi)
            product_from_rxn_without_mapping.append(canon_smi)

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
                    reagents.append(r_clean)
        return reactants, reagents, mapped_products, mapped_rxn, non_smiles_names_list

    @staticmethod
    def rxn_input_extractor(
        rxn: ord_reaction_pb2.Reaction,
        reactants: REACTANTS,
        reagents: REAGENTS,
        solvents: SOLVENTS,
        catalysts: CATALYSTS,
        products: PRODUCTS,
        is_mapped_rxn: bool = False,
    ) -> typing.Tuple[
        REACTANTS,
        REAGENTS,
        SOLVENTS,
        CATALYSTS,
        PRODUCTS,
        typing.List[MOLECULE_IDENTIFIER],
    ]:
        # loop through the keys in the 'dict' style data struct
        non_smiles_names_list = (
            []
        )  # we dont require information on state of this object so can create a new one
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
                    if rxn_role == 1:  # reactant
                        # reactants += [smiles]
                        # we already added reactants from mapped rxn
                        # So instead I'll add it to the reagents list
                        # A lot of the reagents seem to have been misclassified as reactants
                        # I just need to remember to remove reagents that already appear as reactants
                        #   when I do cleaning

                        if is_mapped_rxn:
                            reagents += [r for r in smiles.split(".")]
                        else:
                            reactants += [r for r in smiles.split(".")]
                    elif rxn_role == 2:  # reagent
                        reagents += [r for r in smiles.split(".")]
                    elif rxn_role == 3:  # solvent
                        # solvents += [smiles] # I initially tried to let the solvents stay together, but actually it's better to split them up
                        # Examples like CO.O should probably be split into CO and O
                        solvents += [r for r in smiles.split(".")]
                    elif rxn_role == 4:  # catalyst
                        # catalysts += [smiles] same as solvents
                        catalysts += [r for r in smiles.split(".")]
                    elif rxn_role in [5, 6, 7]:
                        # 5=workup, 6=internal standard, 7=authentic standard. don't care about these
                        continue
                    elif rxn_role == 8:  # product
                        products += [r for r in smiles.split(".")]
                    # there are typically no products recorded in rxn_role == 8, they're all stored in "outcomes"
            except IndexError:
                pass
        return (
            list(set(reactants)),
            list(set(reagents)),
            list(set(solvents)),
            list(set(catalysts)),
            list(set(products)),
            non_smiles_names_list,
        )

    @staticmethod
    def extract_marked_p_and_yields(
        rxn: ord_reaction_pb2.Reaction, marked_products: PRODUCTS
    ) -> typing.Tuple[PRODUCTS, YIELDS, typing.List[MOLECULE_IDENTIFIER]]:
        # products & yield
        yields = []
        non_smiles_names_list = []
        products_obj = rxn.outcomes[0].products
        for marked_product in products_obj:
            y1 = np.nan
            y2 = np.nan
            try:
                identifiers = marked_product.identifiers
                (
                    product_smiles,
                    non_smiles_names_list_additions,
                ) = OrdExtractor.find_smiles(identifiers)

                print(non_smiles_names_list_additions)
                non_smiles_names_list += non_smiles_names_list_additions
                measurements = marked_product.measurements
                for measurement in measurements:
                    if measurement.details == "PERCENTYIELD":
                        y1 = measurement.percentage.value
                    elif measurement.details == "CALCULATEDPERCENTYIELD":
                        y2 = measurement.percentage.value
                marked_products.append(product_smiles)
                if not np.isnan(y1):
                    yields.append(y1)
                elif not np.isnan(y2):
                    yields.append(y2)
                else:
                    yields.append(np.nan)
            except IndexError:
                continue

        return marked_products, yields, non_smiles_names_list

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
                return rxn.conditions.temperature.setpoint.units

            elif temp_unit == 2:  # fahrenheit
                f = rxn.conditions.temperature.setpoint.units
                c = (f - 32) * 5 / 9
                return c

            elif temp_unit == 3:  # kelvin
                k = rxn.conditions.temperature.setpoint.units
                c = k - 273.15
                return c
            elif temp_unit == 0:
                if temp_unit == 0:  # unspecified
                    # instead of using the setpoint, use the control type
                    # temperatures are in celcius
                    temp_control_type = rxn.conditions.temperature.control.type
                    if temp_control_type == 2:  # AMBIENT
                        return 25
                    elif temp_control_type == 6:  # ICE_BATH
                        return 0
                    elif temp_control_type == 9:  # DRY_ICE_BATH
                        return -78.5
                    elif temp_control_type == 11:  # LIQUID_NITROGEN
                        return -196
        except IndexError:
            pass
        return None  # No temperature found

    @staticmethod
    def rxn_time_extractor(rxn: ord_reaction_pb2.Reaction) -> typing.Optional[float]:
        try:
            if rxn.outcomes[0].reaction_time.units == 1:  # hour
                return rxn.outcomes[0].reaction_time.value
            elif rxn.outcomes[0].reaction_time.units == 3:  # seconds
                s = rxn.outcomes[0].reaction_time.value
                h = s / 3600
                return h
            elif rxn.outcomes[0].reaction_time.units == 2:  # minutes
                m = rxn.outcomes[0].reaction_time.value
                h = m / 60
                return h
            elif rxn.outcomes[0].reaction_time.units == 4:  # day
                d = rxn.outcomes[0].reaction_time.value
                h = d * 24
                return h
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
    def merge_marked_mapped_products(
        mapped_p_clean: PRODUCTS, marked_p_clean: PRODUCTS, yields: YIELDS
    ) -> typing.Tuple[PRODUCTS, YIELDS]:
        # merge marked and mapped products with a yield
        products = []
        mapped_yields = []
        for mapped_p in mapped_p_clean:
            added = False
            for ii, marked_p in enumerate(marked_p_clean):
                if (mapped_p == marked_p) and (mapped_p not in products):
                    products.append(mapped_p)
                    mapped_yields.append(
                        yields[ii]
                    )  # I have to do it like this, since I trust the mapped product more, but the yield is associated with the marked product
                    added = True
                    break

            if not added and (mapped_p not in products):
                products.append(mapped_p)
                mapped_yields.append(np.nan)
        return products, mapped_yields

    @staticmethod
    def merge_to_agents(
        catalysts: CATALYSTS,
        solvents: SOLVENTS,
        reagents: REACTANTS,
        metals: METALS,
        solvents_set: typing.Set[SOLVENT],
    ) -> typing.Tuple[AGENTS, SOLVENTS]:
        agents = (
            catalysts + solvents + reagents
        )  # merge the solvents, reagents, and catalysts into one list
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
        rxn: ord_reaction_pb2.Reaction, manual_replacements_dict: dict
    ) -> typing.Tuple[
        REACTANTS,
        REAGENTS,
        SOLVENTS,
        CATALYSTS,
        PRODUCTS,
        YIELDS,
        TEMPERATURE_CELCIUS,
        RXN_TIME,
        MAPPED_RXN,
        str,
        typing.List[MOLECULE_IDENTIFIER],
    ]:
        # handle rxn inputs: reactants, reagents etc

        # initilise empty
        reactants = []
        reagents = []
        solvents = []
        catalysts = []
        marked_products = []
        mapped_products = []
        rxn_non_smiles_names_list = []

        try:  # to extract info from the mapped reaction
            (
                reactants,
                reagents,
                mapped_products,
                mapped_rxn,
                rxn_non_smiles_names_list,
            ) = OrdExtractor.extract_info_from_rxn_str(rxn)
        except ValueError:
            # we don't have a mapped reaction, so we have to just trust the labelled reactants, agents, and products
            mapped_rxn = []

        is_mapped = rxn.identifiers[0].is_mapped
        (
            reactants,
            reagents,
            solvents,
            catalysts,
            mapped_products,
            non_smiles_names_list_additions,
        ) = OrdExtractor.rxn_input_extractor(
            rxn,
            reactants,
            reagents,
            solvents,
            catalysts,
            mapped_products,
            is_mapped,
        )
        rxn_non_smiles_names_list += non_smiles_names_list_additions

        # marked products and yields extraction

        (
            marked_products,
            yields,
            non_smiles_names_list_additions,
        ) = OrdExtractor.extract_marked_p_and_yields(rxn, marked_products)
        rxn_non_smiles_names_list += non_smiles_names_list_additions



        # extract temperature
        temperature = OrdExtractor.temperature_extractor(rxn)

        # extract rxn_time
        rxn_time = OrdExtractor.rxn_time_extractor(rxn)

        # clean the smiles

        # remove molecules that are integers
        reactants = [x for x in reactants if not (x.isdigit())]
        reagents = [x for x in reagents if not (x.isdigit())]
        solvents = [x for x in solvents if not (x.isdigit())]
        catalysts = [x for x in catalysts if not (x.isdigit())]

        def canonicalise_and_get_non_smiles_names(
            mole_id_list: REACTANTS | REAGENTS | SOLVENTS | CATALYSTS,
            is_mapped: bool = False,
        ) -> typing.Tuple[
            REACTANTS | REAGENTS | SOLVENTS | CATALYSTS,
            typing.List[MOLECULE_IDENTIFIER],
        ]:
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

        (
            reactants,
            non_smiles_names_list_additions,
        ) = canonicalise_and_get_non_smiles_names(
            mole_id_list=reactants, is_mapped=True
        )
        rxn_non_smiles_names_list += non_smiles_names_list_additions


        (
            reagents,
            non_smiles_names_list_additions,
        ) = canonicalise_and_get_non_smiles_names(
            mole_id_list=reagents, is_mapped=False
        )
        rxn_non_smiles_names_list += non_smiles_names_list_additions

        (
            solvents,
            non_smiles_names_list_additions,
        ) = canonicalise_and_get_non_smiles_names(
            mole_id_list=solvents, is_mapped=False
        )
        rxn_non_smiles_names_list += non_smiles_names_list_additions

        (
            catalysts,
            non_smiles_names_list_additions,
        ) = canonicalise_and_get_non_smiles_names(
            mole_id_list=catalysts, is_mapped=False
        )
        rxn_non_smiles_names_list += non_smiles_names_list_additions



        # reactants = [self.clean_smiles(smi, is_mapped=True) for smi in reactants]
        # reagents = [self.clean_smiles(smi) for smi in reagents]
        # solvents = [self.clean_smiles(smi) for smi in solvents]
        # catalysts = [self.clean_smiles(smi) for smi in catalysts]

        # Apply the manual_replacements_dict to the reagents, solvents, and catalysts
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

        reagents = [r for r in reagents if r not in reactants]
        solvents = [s for s in solvents if s not in reactants]
        catalysts = [c for c in catalysts if c not in reactants]

        # products logic
        # handle the products
        # for each row, I will trust the mapped product more
        # loop over the mapped products, and if the mapped product exists in the marked product
        # add the yields, else simply add smiles and np.nan

        # canon and remove mapped info from products

        (
            mapped_p_clean,
            non_smiles_names_list_additions,
        ) = canonicalise_and_get_non_smiles_names(
            mole_id_list=mapped_products, is_mapped=True
        )
        rxn_non_smiles_names_list += non_smiles_names_list_additions


        (
            marked_p_clean,
            non_smiles_names_list_additions,
        ) = canonicalise_and_get_non_smiles_names(
            mole_id_list=marked_products, is_mapped=False
        )
        rxn_non_smiles_names_list += non_smiles_names_list_additions


        # mapped_p_clean = [
        #     self.clean_smiles(p, is_mapped=True) for p in mapped_products
        # ]
        # marked_p_clean = [self.clean_smiles(p) for p in marked_products]
        # What if there's a marked product that only has the correct name, but not the smiles?

        if is_mapped:
            # merge marked and mapped products with yield info
            # This is complicated because we trust the mapped reaction more, however, the yields are associated with the marked/labelled products
            products, yields = OrdExtractor.merge_marked_mapped_products(
                mapped_p_clean, marked_p_clean, yields
            )
        else:
            products, yields = marked_p_clean, yields

        procedure_details = rxn.notes.procedure_details


        return (
            reactants,
            reagents,
            solvents,
            catalysts,
            products,
            yields,
            temperature,
            rxn_time,
            mapped_rxn,
            procedure_details,
            rxn_non_smiles_names_list,
        )

    def build_rxn_lists(
        self,
    ) -> typing.Tuple[
        typing.List[MAPPED_RXN],
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
        mapped_rxn_all = []
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
                reagents,
                solvents,
                catalysts,
                products,
                yields,
                temperature,
                rxn_time,
                mapped_rxn,
                procedure_details,
                rxn_non_smiles_names_list_additions,
            ) = OrdExtractor.handle_reaction_object(
                rxn, manual_replacements_dict=self.manual_replacements_dict
            )
            rxn_non_smiles_names_list += rxn_non_smiles_names_list_additions

            if set(reactants) != set(
                products
            ):  # If the reactants and products are the same, then we don't want to add this reaction to our dataset
                mapped_rxn_all.append(mapped_rxn)
                reactants_all.append(reactants)
                products_all.append(products)
                yields_all.append(yields)
                temperature_all.append(temperature)
                rxn_time_all.append(rxn_time)
                procedure_details_all.append(procedure_details)

                # Finally we also need to add the agents
                if self.merge_cat_solv_reag == True:
                    agents, solvents = OrdExtractor.merge_to_agents(
                        catalysts, solvents, reagents, self.metals, self.solvents_set
                    )
                    agents_all.append(agents)
                    solvents_all.append(solvents)
                    reagents_all.append(None)
                    catalysts_all.append(None)
                else:
                    agents_all.append(None)
                    solvents_all.append(list(set(solvents)))
                    reagents_all.append(list(set(reagents)))
                    catalysts_all.append(list(set(catalysts)))

        return (
            mapped_rxn_all,
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
            "mapped_rxn_",
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
