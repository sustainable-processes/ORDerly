import logging
import typing
import pathlib
import dataclasses

import numpy as np
import pandas as pd

from ord_schema import message_helpers as ord_message_helpers
from ord_schema.proto import dataset_pb2 as ord_dataset_pb2

from rdkit import Chem as rdkit_Chem
from rdkit.rdBase import BlockLogs as rdkit_BlockLogs

import orderly.extract.defaults

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
    solvents_set: typing.Set[str]
    filename: typing.Optional[str] = None
    contains_substring: typing.Optional[str] = None  # None or uspto
    inverse_contains_substring: bool = False

    def __post_init__(self):
        LOG.debug(f"Extracting data from {self.ord_file_path}")
        self.data = ord_message_helpers.load_message(
            str(self.ord_file_path), ord_dataset_pb2.Dataset
        )
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
                LOG.debug(f"No file name for dataset so using {self.data.dataset_id=}")
                self.filename = self.data.dataset_id

        self.names_list = self.full_df = None
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

        self.names_list = []
        self.full_df = self.build_full_df()
        LOG.debug(f"Got data from {self.ord_file_path}: {self.filename}")

    def find_smiles(self, identifiers):
        _ = rdkit_BlockLogs()
        for i in identifiers:
            if i.type == 2:
                smiles = self.clean_smiles(i.value)
                return smiles
        for ii in identifiers:  # if there's no smiles, return the name
            if ii.type == 6:
                name = ii.value
                self.names_list += [name]
                return name
        return None

    def clean_mapped_smiles(self, smiles):
        _ = rdkit_BlockLogs()
        # remove mapping info and canonicalsie the smiles at the same time
        # converting to mol and back canonicalises the smiles string
        try:
            m = rdkit_Chem.MolFromSmiles(smiles)
            for atom in m.GetAtoms():
                atom.SetAtomMapNum(0)
            cleaned_smiles = rdkit_Chem.MolToSmiles(m)
            return cleaned_smiles
        except AttributeError:
            self.names_list += [smiles]
            return smiles

    def clean_smiles(self, smiles):
        _ = rdkit_BlockLogs()
        # remove mapping info and canonicalsie the smiles at the same time
        # converting to mol and back canonicalises the smiles string
        try:
            cleaned_smiles = rdkit_Chem.CanonSmiles(smiles)
            return cleaned_smiles
        except:
            self.names_list += [smiles]
            return smiles

    # its probably a lot faster to sanitise the whole thing at the end
    # NB: And create a hash map/dict

    def build_rxn_lists(
        self, metals: typing.Optional[typing.List[str]] = None
    ) -> typing.Tuple[
        typing.List,
        typing.List,
        typing.List,
        typing.List,
        typing.List,
        typing.List,
        typing.List,
        typing.List,
        typing.List,
        typing.List,
        typing.List,
    ]:
        if metals is None:
            metals = orderly.extract.defaults.get_metals_list()

        mapped_rxn_all = []
        reactants_all = []
        reagents_all = []
        agents_all = []
        products_all = []
        solvents_all = []
        catalysts_all = []

        temperature_all = []
        rxn_times_all = []

        yields_all = []
        procedure_details_all = []

        for i in range(len(self.data.reactions)):
            rxn = self.data.reactions[i]
            # handle rxn inputs: reactants, reagents etc
            reactants = []
            reagents = []
            solvents = []
            catalysts = []
            marked_products = []
            mapped_products = []
            products = []
            not_mapped_products = []

            temperatures = []
            rxn_times = []

            yields = []
            mapped_yields = []

            # Add procedure_details
            procedure_details = [rxn.notes.procedure_details]
            procedure_details_all += [procedure_details]

            # if reaction has been mapped, get reactant and product from the mapped reaction
            # Actually, we should only extract data from reactions that have been mapped
            if len(self.data.reactions[i].identifiers) == 0:
                LOG.debug("skipping, reactions have no identifiers")
                continue

            is_mapped = self.data.reactions[i].identifiers[0].is_mapped
            if is_mapped:
                mapped_rxn_extended_smiles = self.data.reactions[i].identifiers[0].value
                mapped_rxn = mapped_rxn_extended_smiles.split(" ")[0] # this is to get rid of the extended smiles info

                reactant, reagent, mapped_product = mapped_rxn.split(">")

                reactants += [r for r in reactant.split(".")] #we're trusting the labelling of reagents and reactants here
                reagents += [r for r in reagent.split(".")] # Maybe in the future we'd add atom mapping to determine which is which

                for p in mapped_product.split("."):
                    if "[" in p and "]" in p and ":" in p:
                        mapped_products += [p]

                    else:
                        not_mapped_products += [p]

                # loop through the keys in the 'dict' style data struct
                for key in rxn.inputs:
                    try:
                        components = rxn.inputs[key].components
                        for component in components:
                            rxn_role = component.reaction_role  # rxn role
                            identifiers = component.identifiers
                            smiles = self.find_smiles(identifiers)
                            if rxn_role == 1:  # reactant
                                # reactants += [smiles]
                                # we already added reactants from mapped rxn
                                # So instead I'll add it to the reagents list
                                # A lot of the reagents seem to have been misclassified as reactants
                                # I just need to remember to remove reagents that already appear as reactants
                                #   when I do cleaning

                                reagents += [r for r in smiles.split(".")]
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
                            # elif rxn_role ==8: #product
                            #     #products += [smiles]
                            # there are no products recorded in rxn_role == 8, they're all stored in "outcomes"
                    except IndexError:
                        continue

                # temperature
                try:
                    # first look for the temperature as a number
                    temp_unit = rxn.conditions.temperature.setpoint.units

                    if temp_unit == 1:  # celcius
                        temperatures += [rxn.conditions.temperature.setpoint.units]

                    elif temp_unit == 2:  # fahrenheit
                        f = rxn.conditions.temperature.setpoint.units
                        c = (f - 32) * 5 / 9
                        temperatures += [c]

                    elif temp_unit == 3:  # kelvin
                        k = rxn.conditions.temperature.setpoint.units
                        c = k - 273.15
                        temperatures += [c]
                    elif temp_unit == 0:
                        if temp_unit == 0:  # unspecified
                            # instead of using the setpoint, use the control type
                            # temperatures are in celcius
                            temp_control_type = rxn.conditions.temperature.control.type
                            if temp_control_type == 2:  # AMBIENT
                                temperatures += [25]
                            elif temp_control_type == 6:  # ICE_BATH
                                temperatures += [0]
                            elif temp_control_type == 9:  # DRY_ICE_BATH
                                temperatures += [-78.5]
                            elif temp_control_type == 11:  # LIQUID_NITROGEN
                                temperatures += [-196]
                except IndexError:
                    continue

                # rxn time
                try:
                    if rxn.outcomes[0].reaction_time.units == 1:  # hour
                        rxn_times += [rxn.outcomes[0].reaction_time.value]
                    elif rxn.outcomes[0].reaction_time.units == 3:  # seconds
                        s = rxn.outcomes[0].reaction_time.value
                        h = s / 3600
                        rxn_times += [h]
                    elif rxn.outcomes[0].reaction_time.units == 2:  # minutes
                        m = rxn.outcomes[0].reaction_time.value
                        h = m / 60
                        rxn_times += [h]
                    elif rxn.outcomes[0].reaction_time.units == 4:  # day
                        d = rxn.outcomes[0].reaction_time.value
                        h = d * 24
                        rxn_times += [h]
                except IndexError:
                    continue

                # products & yield
                products_obj = rxn.outcomes[0].products
                y1 = np.nan
                y2 = np.nan
                for marked_product in products_obj:
                    try:
                        identifiers = marked_product.identifiers
                        product_smiles = self.find_smiles(identifiers)
                        measurements = marked_product.measurements
                        for measurement in measurements:
                            if measurement.details == "PERCENTYIELD":
                                y1 = measurement.percentage.value
                            elif measurement.details == "CALCULATEDPERCENTYIELD":
                                y2 = measurement.percentage.value
                        # marked_products += [(product_smiles, y1, y2)]
                        marked_products += [product_smiles]
                        if y1 == y1:
                            yields += [y1]
                        elif y2 == y2:
                            yields += [y2]
                        else:
                            yields += [np.nan]
                    except IndexError:
                        continue

            # clean the smiles

            # remove molecules that are integers
            reagents = [x for x in reagents if not (x.isdigit())]
            catalysts = [x for x in catalysts if not (x.isdigit())]

            reactants = [self.clean_mapped_smiles(smi) for smi in reactants]

            reagents = [self.clean_smiles(smi) for smi in reagents]
            solvents = [self.clean_smiles(smi) for smi in solvents]
            catalysts = [self.clean_smiles(smi) for smi in catalysts]

            # Apply the manual_replacements_dict to the reagents, solvents, and catalysts
            reagents = [
                x
                for x in pd.Series(reagents, dtype=pd.StringDtype())
                .map(
                    lambda x: self.manual_replacements_dict.get(x, x),
                    na_action="ignore",
                )
                .tolist()
                if x is not None
            ]

            solvents = [
                x
                for x in pd.Series(reagents, dtype=pd.StringDtype())
                .map(
                    lambda x: self.manual_replacements_dict.get(x, x),
                    na_action="ignore",
                )
                .tolist()
                if x is not None
            ]
            catalysts = [
                x
                for x in pd.Series(reagents, dtype=pd.StringDtype())
                .map(
                    lambda x: self.manual_replacements_dict.get(x, x),
                    na_action="ignore",
                )
                .tolist()
                if x is not None
            ]

            # Split out any instances of a . in the smiles strings
            reagents = [
                substring for reagent in reagents for substring in reagent.split(".")
            ]
            solvents = [
                substring for solvent in solvents for substring in solvent.split(".")
            ]
            catalysts = [
                substring for catalyst in catalysts for substring in catalyst.split(".")
            ]

            # if reagent appears in reactant list, remove it
            # Since we're technically not sure whether something is a reactant (contributes atoms) or a reagent/solvent/catalyst (does not contribute atoms), it's probably more cautious to remove molecules that appear in both lists from the reagents/solvents/catalysts list rather than the reactants list
            reagents = [reagent for reagent in reagents if reagent not in reactants]
            solvents = [solvent for solvent in solvents if solvent not in reactants]
            catalysts = [catalyst for catalyst in catalysts if catalyst not in reactants]

            # products logic
            # handle the products
            # for each row, I will trust the mapped product more
            # loop over the mapped products, and if the mapped product exists in the marked product
            # add the yields, else simply add smiles and np.nan

            # canon and remove mapped info from products
            mapped_p_clean = [self.clean_mapped_smiles(p) for p in mapped_products]
            marked_p_clean = [self.clean_smiles(p) for p in marked_products]
            # What if there's a marked product that only has the correct name, but not the smiles?

            for mapped_p in mapped_p_clean:
                added = False
                for ii, marked_p in enumerate(marked_p_clean):
                    if mapped_p == marked_p and mapped_p not in products:
                        products += [mapped_p]
                        mapped_yields += [yields[ii]]
                        added = True
                        break

                if not added and mapped_p not in products:
                    products += [mapped_p]
                    mapped_yields += [np.nan]

            if set(reactants) != set(
                products
            ):  # If the reactants and products are the same, then we don't want to add this reaction to our dataset
                mapped_rxn_all += [mapped_rxn]
                reactants_all += [reactants]

                products_all += [products]
                yields_all += [mapped_yields]

                temperature_all = [temperatures]

                rxn_times_all += [rxn_times]

                # Finally we also need to add the agents
                if self.merge_cat_solv_reag == True:
                    agents = (
                        catalysts + solvents + reagents
                    )  # merge the solvents, reagents, and catalysts into one list
                    agents_set = set(agents)  # this includes the solvnts

                    # build two new lists, one with the solvents, and one with the reagents+catalysts
                    # Create a new set of solvents from agents_set
                    solvents = agents_set.intersection(self.solvents_set)

                    # Remove the solvents from agents_set
                    agents = agents_set.difference(solvents)

                    # I think we should add some ordering to the agents
                    # What if we first order them alphabetically, and afterwards by putting the metals first in the list

                    agents = sorted(list(agents))
                    solvents = sorted(list(solvents))

                    # Ideally we'd order the agents, so we have the catalysts (metal centre) first, then the ligands, then the bases and finally any reagents
                    # We don't have a list of catalysts, and it's not straight forward to figure out if something is a catalyst or not (both chemically and computationally)
                    # Instead, let's move all agents that contain a metal centre to the front of the list

                    agents = [
                        agent
                        for agent in agents
                        if any(metal in agent for metal in metals)
                    ] + [
                        agent
                        for agent in agents
                        if not any(metal in agent for metal in metals)
                    ]

                    agents_all += [agents]
                    solvents_all += [solvents]
                else:
                    solvents_all += [list(set(solvents))]
                    reagents_all += [list(set(reagents))]
                    catalysts_all += [list(set(catalysts))]

        return (
            mapped_rxn_all,
            reactants_all,
            agents_all,
            reagents_all,
            solvents_all,
            catalysts_all,
            temperature_all,
            rxn_times_all,
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
            column_headers += [base_string + str(i)]
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
