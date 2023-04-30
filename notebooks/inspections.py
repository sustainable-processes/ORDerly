# type: ignore
# %%
set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn: bool = True
remove_rxn_with_unresolved_names: bool = False
set_unresolved_names_to_none: bool = False


# %%
from rdkit import Chem as rdkit_Chem


def is_mapped(rxn_str):
    """
    Check if a reaction string is mapped using RDKit.
    """
    reactants, _, _ = rxn_str.split(">")
    reactants = reactants.split(".")
    for r in reactants:
        mol = rdkit_Chem.MolFromSmiles(r)
        if mol != None:
            if any(atom.HasProp("molAtomMapNumber") for atom in mol.GetAtoms()):
                return True
    return False


# %%
is_mapped(
    "[S:1](=[O:4])(=[O:3])=[O:2].[S:5](=[O:9])(=[O:8])([OH:7])[OH:6]>>[OH:8][S:5]([OH:9])(=[O:7])=[O:6].[O:2]=[S:1](=[O:4])=[O:3]"
)

# %%
is_mapped("CCC.C>O>CCC")


# %% [markdown]
# # Test merge ions to salt

# %%
from rdkit import Chem


# %%
def _merge_ions_to_salt(mole_id_list):
    """If there's just 1 positive and 1 negative ion, merge these to a salt. E.g. [Na+].[OH-] becomes O[Na]"""

    mole_id_list_with_ions_merged = []
    for smiles in mole_id_list:
        mol = Chem.MolFromSmiles(smiles)
        # Identify the anions and cations in the molecule
        anions = [
            atom.GetIdx() for atom in mol.GetAtoms() if atom.GetFormalCharge() < 0
        ]
        cations = [
            atom.GetIdx() for atom in mol.GetAtoms() if atom.GetFormalCharge() > 0
        ]
        # If there is exactly one anion and one cation, merge them to form a salt
        if len(anions) == 1 and len(cations) == 1:
            anion_idx = anions[0]
            cation_idx = cations[0]
            # Create a new molecule with the anion and cation bonded together
            new_mol = Chem.RWMol(mol)
            new_mol.AddBond(anion_idx, cation_idx, order=Chem.rdchem.BondType.SINGLE)
            # Set the formal charge of the new molecule to zero
            new_mol.GetAtomWithIdx(anion_idx).SetFormalCharge(0)
            new_mol.GetAtomWithIdx(cation_idx).SetFormalCharge(0)
            # Convert the new molecule back to a SMILES string and add it to the list
            new_smiles = Chem.MolToSmiles(new_mol.GetMol())
            mole_id_list_with_ions_merged.append(new_smiles)
        else:
            mole_id_list_with_ions_merged.append(smiles)
    return mole_id_list_with_ions_merged


# %%
def merge_ions_to_salt_func(
    mole_id_list: REACTANTS | AGENTS | REAGENTS | SOLVENTS | CATALYSTS | PRODUCTS,
) -> REACTANTS | AGENTS | REAGENTS | SOLVENTS | CATALYSTS | PRODUCTS:
    """If there's just 1 positive and 1 negative ion, merge these to a salt. E.g. [Na+].[OH-] becomes O[Na]"""
    assert isinstance(mole_id_list, list)

    mole_id_list_with_ions_merged = []
    for smiles in mole_id_list:
        mol = Chem.MolFromSmiles(smiles)
        # Identify the anions and cations in the molecule
        anions = [
            atom.GetIdx() for atom in mol.GetAtoms() if atom.GetFormalCharge() < 0
        ]
        cations = [
            atom.GetIdx() for atom in mol.GetAtoms() if atom.GetFormalCharge() > 0
        ]
        # If there is exactly one anion and one cation, merge them to form a salt
        if len(anions) == 1 and len(cations) == 1:
            anion_idx = anions[0]
            cation_idx = cations[0]
            # Create a new molecule with the anion and cation bonded together
            new_mol = Chem.RWMol(mol)
            new_mol.AddBond(anion_idx, cation_idx, order=Chem.rdchem.BondType.SINGLE)
            # Set the formal charge of the new molecule to zero
            new_mol.GetAtomWithIdx(anion_idx).SetFormalCharge(0)
            new_mol.GetAtomWithIdx(cation_idx).SetFormalCharge(0)
            # Convert the new molecule back to a SMILES string and add it to the list
            new_smiles = Chem.MolToSmiles(new_mol.GetMol())
            mole_id_list_with_ions_merged.append(new_smiles)
        else:
            mole_id_list_with_ions_merged.append(smiles)
    return mole_id_list_with_ions_merged


# %%
_merge_ions_to_salt(["[Na+].[OH-].C", "CC", "[Pd+]"])

# %%


# %%


# %%


# %%


# %% [markdown]
# # inspect df

# %%
import orderly

train, val, test = orderly.get_uspto_rxn_class_split()

# %%
# "ord_dataset-0bf72e95d80743729fdbb8b57a4bc0c6",
# 0,
# "[CH3:1][C:2]1([CH3:10])[CH:8]2[CH2:9][CH:3]1[CH2:4][CH2:5][C:6]2=O.[NH:11]1[CH2:15][CH2:14][CH2:13][CH2:12]1.C1(C)C=CC(S(O)(=O)=O)=CC=1.O>C1C=CC=CC=1>[CH3:1][C:2]1([CH3:10])[CH:8]2[CH2:9][CH:3]1[CH2:4][CH:5]=[C:6]2[N:11]1[CH2:15][CH2:14][CH2:13][CH2:12]1"

# ["C1CCNC1","CC1(C)C2CCC(=O)C1C2",],

# ["Cc1ccc(S(=O)(=O)O)cc1", "O", "c1ccccc1"],

# ["CC1(C)C2CC=C(N3CCCC3)C1C2"],


# %%
import pandas as pd

# read in data
path = "/Users/dsw46/Library/CloudStorage/OneDrive-UniversityofCambridge/Datasets/orderly_defaults_data.parquet"

df = pd.read_parquet(path)

# %%
df.columns

# %%
df2 = (
    df[
        [
            "reactant_0",
            "product_0",
            "solvent_0",
            "agent_0",
            "temperature_0",
            "rxn_time_0",
        ]
    ]
    .iloc[1:6]
    .copy()
)

# %%
df2["agent_0"].iloc[0] = "other"

# %%
df2

# %% [markdown]
# # Removing try catch blocks from extractor

# %%
# Import modules
import ord_schema
from ord_schema import message_helpers, validations
from ord_schema.proto import dataset_pb2
from rdkit.rdBase import BlockLogs as rdkit_BlockLogs


# %%
def find_smiles(
    identifiers,
):
    """
    Search through the identifiers to return a smiles string, if this doesn't exist, search to return the English name, and if this doesn't exist, return None
    """
    non_smiles_names_list = (
        []
    )  # we dont require information on state of this object so can create a new one
    _ = rdkit_BlockLogs()
    for i in identifiers:
        if i.type == 2:
            canon_smi = i.value
            return canon_smi, non_smiles_names_list
    for ii in identifiers:  # if there's no smiles, return the name
        if ii.type == 6:
            name = ii.value
            non_smiles_names_list.append(name)
            return name, non_smiles_names_list
    return None, non_smiles_names_list


def rxn_input_extractor(rxn):
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
            smiles, non_smiles_names_list_additions = find_smiles(identifiers)
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
    return (
        reactants,
        reagents,
        solvents,
        catalysts,
        products,
        non_smiles_names_list,
    )


# %%
# Load Dataset message
pb = "data/ord//02/ord_dataset-02ee2261663048188cf6d85d2cc96e3f.pb.gz"
pb = "orderly/data/ord_test_data/00/ord_dataset-00005539a1e04c809a9a78647bea649c.pb.gz"
data = message_helpers.load_message(pb, dataset_pb2.Dataset)

# %%
data.reactions[0].outcomes[0].products[0].measurements

# %%
import pandas as pd

# %%
data.reactions[0].provenance.experiment_start.value

# %%


# %%
i = 0
d = data.reactions[i].provenance.experiment_start.value
print(d)
type(pd.to_datetime(d, format="%m/%d/%Y"))


# %%
import pandas as pd

df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})

# %%
df["b"].dropna()

# %%
pd.Timestamp(data.reactions[0].provenance.experiment_start.value)

# %%
for i in range(10):
    print(data.reactions[i].provenance.experiment_start.value)

# %% [markdown]
# ## First try: simply remove

# %% [markdown]
# ## Second try: add catch for measurements = None, and remove yield if len(products_list) != 1

# %%
# before
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

# %%
# before
for product in products_obj:
    y = None
    identifiers = product.identifiers
    (
        product_smiles,
        non_smiles_names_list_additions,
    ) = OrdExtractor.find_smiles(identifiers)

    non_smiles_names_list += non_smiles_names_list_additions
    measurements = product.measurements
    if measurements is not None:
        for measurement in measurements:
            if measurement.type == 3:  # YIELD
                y = float(measurement.percentage.value)
                y = round(y, 2)
    # people sometimes report a product such as '[Na+].[Na+].[O-]B1OB2OB([O-])OB(O1)O2' and then only report one yield, this is a problem...
    # We'll resolve this by moving the longest smiles string to the front of the list, then appending the yield to the front of the list, and padding with None to ensure that the lists are the same length

    # split the product string by dot and sort by descending length
    # ASSUMPTION: The largest product is the one with the yield
    product_list = sorted(product_smiles.split("."), key=len, reverse=True)

    # create a list of the same length as product_list with y as the first value and None as the other values
    y_list = [y] + [None] * (len(product_list) - 1)

    products += product_list
    yields += y_list


# %%
a = [1]
a += [1, 2, 3]
a

# %% [markdown]
# # Inspect clean df

# %%
import pandas as pd

df = pd.read_parquet("data/orderly/orderly_ord.parquet")

# %%
# Define the list of columns to check
columns_to_check = [
    col
    for col in df.columns
    if col.startswith(("agent", "solvent", "reagent", "catalyst"))
]

value_counts = df.value_counts(subset=columns_to_check)

# %%
cutoff = 5
# Define the list of columns to check
columns_to_check = [
    col
    for col in df.columns
    if col.startswith(("agent", "solvent", "reagent", "catalyst"))
]

# Initialize a list to store the results
results = []

# Loop through the columns
for col in columns_to_check:
    # Get the value counts for the column
    results += [df[col].value_counts()]

total_value_counts = pd.concat(results, axis=0, sort=True).groupby(level=0).sum()
total_value_counts = total_value_counts.drop("other")
total_value_counts = total_value_counts.sort_values(ascending=True)

# %%
total_value_counts[0]

# %%
pd.DataFrame(
    {
        "reactant_0": ["B", "A", "F", "A"],
        "reactant_1": ["D", "A", "G", "B"],
        "product_0": ["C", "A", "E", "A"],
        "product_1": ["E", "G", "C", "H"],
        "agent_0": ["D", "F", "D", "B"],
        "agent_1": ["C", "E", "G", "A"],
        "solvent_0": ["E", "B", "G", "C"],
        "solvent_1": ["C", "D", "B", "G"],
        "solvent_2": ["D", "B", "F", "G"],
    }
)

{
    "reactant_0": ["B", "F"],
    "reactant_1": ["D", "G"],
    "product_0": ["C", "E"],
    "product_1": ["E", "C"],
    "agent_0": ["D", "D"],
    "agent_1": ["C", "G"],
    "solvent_0": ["E", "G"],
    "solvent_1": ["C", "B"],
    "solvent_2": ["D", "F"],
}


# %%
def _remove_rare_molecules(
    df, columns_to_transform, value_counts, min_frequency_of_occurrence
):
    """
    Removes rows with rare values in specified columns.
    """
    # Get the indices of rows where the column contains a rare value
    rare_values = value_counts[value_counts < min_frequency_of_occurrence].index
    index_union = None

    for col in columns_to_transform:
        mask = df[col].isin(rare_values)
        rare_indices = df.loc[mask].index
        # Remove the rows with rare values
        df = df.drop(rare_indices)
    return df


# %%
df = pd.DataFrame(
    {
        "reactant_0": ["B", "A", "F", "A"],
        "reactant_1": ["D", "A", "G", "B"],
        "product_0": ["C", "A", "E", "A"],
        "product_1": ["E", "G", "C", "H"],
        "agent_0": ["D", "F", "D", "B"],
        "agent_1": ["C", "E", "G", "A"],
        "solvent_0": ["E", "B", "G", "C"],
        "solvent_1": ["C", "D", "B", "G"],
        "solvent_2": ["D", "B", "F", "G"],
    }
)

columns_to_transform = ["agent_0", "agent_1", "solvent_0", "solvent_1", "solvent_2"]
value_counts = pd.Series({"A": 1, "B": 4, "C": 3, "D": 4, "E": 2, "F": 2, "G": 4})
min_frequency_of_occurrence = 2

_remove_rare_molecules(
    df, columns_to_transform, value_counts, min_frequency_of_occurrence
)

# %%
rare_values = ["A"]

mask = df["reactant_1"].isin(rare_values)
rare_indices = df.loc[mask].index

# %%
mask2 = df["reactant_0"].isin(rare_values)
rare_indices2 = df.loc[mask2].index

# %%
mask2

# %%
mask2

# %%
index_union = rare_indices2.union(rare_indices)

# %%
df.drop(index_union)

# %%
df

# %%


# %%
import typing

# %%
A = typing.NewType("A", str)

# %%
A("asd") == "asd"

# %%
{"abc": 3}["abc"]

# %%
{"abc": 3}[A("abc")]

# %%
{A("asd"): 3}["asd"]

# %% [markdown]
# rare_indices

# %%
{
    "reactant_0": ["A"],
    "reactant_1": ["B"],
    "product_0": ["A"],
    "product_1": ["H"],
    "agent_0": ["B"],
    "agent_1": ["A"],
    "solvent_0": ["C"],
    "solvent_1": ["G"],
    "solvent_2": ["G"],
},

# %%
"rxn_str_0".endswith("0")

# %%
"reagent".startswith(("agent", "solvent", "reagent", "catalyst"))


# %%
def check_frequency_of_occurrence(
    series,
    min_frequency_of_occurrence,
    include_other_category,
    map_rare_to_other_threshold,
):
    # series could be df['agent_0'], df['reagent_1'], df['solvent_0'], etc.
    item_frequencies = series[series != "other"].value_counts()

    # Check that the item with the lowest frequency appears at least `min_frequency_of_occurrence` times
    if len(item_frequencies) > 0:
        least_common_frequency = item_frequencies.iloc[-1]
        if include_other_category:
            # If 'other' is included, then the least common item must appear at least `min_frequency_of_occurrence` times
            assert least_common_frequency >= map_rare_to_other_threshold
        else:
            # If 'other' is not included, then the least common item must appear at least `min_frequency_of_occurrence` times
            assert least_common_frequency >= min_frequency_of_occurrence
    else:
        # If there are no items other than 'other', the test passes
        pass


# %%
series = df["agent_0"]
item_frequencies = series.value_counts()

# %% [markdown]
# # Inspect raw ORD file

# %%
# Find the schema here
# https://github.com/open-reaction-database/ord-schema/blob/main/ord_schema/proto/reaction.proto

# %%
# Import modules
import ord_schema
from ord_schema import message_helpers, validations
from ord_schema.proto import dataset_pb2

import math
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import wget

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn import model_selection, metrics
from glob import glob

from tqdm import tqdm

# %%
# Load Dataset message
pb = "data/ord//02/ord_dataset-02ee2261663048188cf6d85d2cc96e3f.pb.gz"
data = message_helpers.load_message(pb, dataset_pb2.Dataset)

# %%
res = [field.name for field in data.DESCRIPTOR.fields]
res
# can do data.name to get the year, e.g. data.name returns 'uspto-grants-2016'
# can do data.dataset_id to get the ord file name, e.g. 'ord_dataset-026684a62f91469db49c7767d16c39fb'

# %%
data.name

# %%
data.reactions[0]

# %%
res = [field.name for field in data.reactions[200].notes.DESCRIPTOR.fields]
res

# %%
res = [field.name for field in data.reactions[200].identifiers[0].DESCRIPTOR.fields]
res

# %%
# Example of reaction where reactant is also a product, and has been mapped: [Li+:23]

"O[CH:2]([CH2:16][CH2:17][CH2:18][CH2:19][CH2:20][CH3:21])[CH2:3][CH2:4][CH2:5][CH2:6][CH2:7][CH2:8][CH2:9][CH2:10][CH2:11][CH2:12][C:13]([OH:15])=[O:14].[OH-:22].[Li+:23].[P].[S]>>[OH:22][CH:12]([CH2:11][CH2:10][CH2:9][CH2:8][CH2:7][CH2:6][CH2:5][CH2:4][CH2:3][CH2:2][CH2:16][CH2:17][CH2:18][CH2:19][CH2:20][CH3:21])[C:13]([O-:15])=[O:14].[Li+:23]"
# This means we can't use search of '[' to determine whether something is a reactant or not


# %%
# Mapped with rxn mapper
# [CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH2:6][CH:7]([CH2:8][CH2:9][CH2:10][CH2:11][CH2:12][CH2:13][CH2:14][CH2:15][CH2:16][CH2:17][C:19](=[O:20])[OH:21])[OH:18].[Li+:22].[OH-].[P].[S]>>[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH2:6][CH2:7][CH2:8][CH2:9][CH2:10][CH2:11][CH2:12][CH2:13][CH2:14][CH2:15][CH2:16][CH:17]([OH:18])[C:19](=[O:20])[O-:21].[Li+:22]

# %%
for i in range(len(data.reactions)):
    try:
        if data.reactions[i].notes.is_sensitive_to_light:
            print(data.reactions[i].notes.is_sensitive_to_light)
    except:
        pass

# %%
for i in range(len(data.reactions)):
    try:
        print(data.reactions[i].notes.offgasses)
    except:
        pass

# %%
for i in range(len(data.reactions)):
    try:
        print(data.reactions[i].notes.safety_notes)
    except:
        pass

# %% [markdown]
# # Inspecting the output of the cleaning

# %% [markdown]
# ## Name Resolution

# %%
from os import listdir
from os.path import isfile, join
import pandas as pd
from tqdm import tqdm
from collections import Counter
import pickle
import os
from rdkit import Chem

# %%
from rdkit import Chem

mol = Chem.MolFromSmiles("CCCCC")

# %%
mol


# %%
def merge_extracted_ords_mol_names():
    # create one big list of all the pickled names
    folder_path = (
        "/Users/dsw46/Projects/chemical-parameter-sharing/data/USPTO/molecule_names/"
    )
    onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    full_lst = []
    for file in tqdm(onlyfiles):
        if file[0] != ".":  # We don't want to try to unpickle .DS_Store
            filepath = folder_path + file
            unpickled_lst = pd.read_pickle(filepath)
            full_lst = full_lst + unpickled_lst

    return full_lst


# %%
names_list = merge_extracted_ords_mol_names()

# %%
len(names_list)

# %%
"same catalyst" in names_list

# %%


# %%
# Count the frequency of each item in the list
item_counts = Counter(names_list)

# Sort the items by frequency in descending order
sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)

# Print the sorted items
for item, count in sorted_items:
    print(f"{item}: {count}")


# %% [markdown]
# # Verify that all the cat, solv and reagents make sense

# %%
import pandas as pd

# %%
# read in data
data_df = pd.read_pickle("data/USPTO/clean_test_split_cat.pkl")

# %%
value_counts = df[columns[0]].value_counts()
for i in range(1, len(columns)):
    value_counts = value_counts.add(df[columns[i]].value_counts(), fill_value=0)

# %%
# Get list of catalysts
catalysts = list(set(list(data_df["catalyst_0"])))
print(len(catalysts))
catalysts

# %%
# Get list of solvents
solvents = list(set(list(data_df["solvent_0"]) + list(data_df["solvent_1"])))
print(len(solvents))
solvents
# O is not a solvent or a catalyst?? Is it a fair reagent to include?? There's also Pd as a solvent lol

# %%
# Get list of reagents
reagents = list(set(list(data_df["reagent_0"]) + list(data_df["reagent_1"])))
print(len(reagents))
reagents

# CCOCC is a solvent???
# There's "C", "N", and "O" as a reagent, this seems wrong

# %% [markdown]
# # Check out the solvents csv

# %%
solvents = pd.read_csv("data/USPTO/solvents.csv", index_col=0)
solvents.loc[375, "smiles"] = "ClP(Cl)Cl"
solvents.loc[405, "smiles"] = "ClS(Cl)=O"
methanol = {"cosmo_name": "methanol", "stenutz_name": "methanol", "smiles": "CO"}
solvents = solvents.append(methanol, ignore_index=True)

# %%
solvents

# %%
has_nan_1 = solvents["stenutz_name"].isnull().any()
has_nan_2 = solvents["cosmo_name"].isnull().any()

# Print the result
print(f"The Series has NaN values: {has_nan_1}, {has_nan_2}")


# %%
def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


# apply the function to the 'smiles' column of the dataframe
solvents["canonical_smiles"] = solvents["smiles"].apply(canonicalize_smiles)

# %%
for i in range(len(solvents)):
    if solvents["smiles"][i] == "O":
        print(i)

# %%
## Check how many of the solvents in the data are actually solvents
s = data_df["solvent_0"]
my_list = solvents["canonical_smiles"]

# Count the number of values in the pd.series that are in my_list
in_list = s.isin(my_list).sum()

# Count the number of values in the pd.series that are not in my_list
not_in_list = (~s.isin(my_list)).sum()

# Count the number of values in the pd.series that are None or NaN
null_values = s.isnull().sum()

print(
    f"{in_list} values are in the list, {len(s)-in_list-null_values} values are not in the list, and {null_values} values are None or NaN"
)

# %%
not_in_list = s[~s.isin(my_list)]

# Sort the values in descending order of frequency
freq_not_in_list = not_in_list.value_counts()

# # Print the values, starting with the most frequent
for value, count in freq_not_in_list.items():
    print(f"{value}: {count}")

# %%
my_list

# %%
len(s)

# %%
len(s.dropna())


# %%
# Inspect pickled data
def merge_extracted_ords():
    # create one big df of all the pickled data
    folder_path = "data/USPTO/extracted_ords/"
    onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    full_df = pd.DataFrame()
    for file in tqdm(onlyfiles[:100]):
        if file[0] != ".":  # We don't want to try to unpickle .DS_Store
            filepath = folder_path + file
            unpickled_df = pd.read_pickle(filepath)
            full_df = pd.concat([full_df, unpickled_df], ignore_index=True)

    return full_df


# %%
df = merge_extracted_ords()

# %%
list(df.columns)

# %%
import pandas as pd

# Example Series and dictionary
reagents = pd.Series(["reagent1", "reagent2", "reagent3", "reagent4"])
replace_dict = {"reagent2": "a.b", "reagent4": "new_reagent4"}

# Apply the dictionary replacement to the Series in-place
reagents.replace(replace_dict, inplace=True)

# Print the resulting Series
print(list(reagents))


# %%
# Example list of reagents
reagents = ["reagent1", "a.b", "reagent3", "new_reagent4"]

# Separate strings with a '.' character into two items and sort alphabetically
reagents_new = [substring for reagent in reagents for substring in reagent.split(".")]

# Print the resulting list
print(reagents_new)


# %%
reag = pd.Series(["reagent1", "reagent2", "reagent3", "reagent4"])
replace_dict = {"reagent2": "a.b", "reagent4": "new_reagent4"}

list((pd.Series(reag)).map(replace_dict))

# %%
reagents_new


# %%
def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)


# %%
def build_replacements():
    molecule_replacements = {}

    # Add a catalyst to the molecule_replacements dict (Done by Alexander)
    molecule_replacements[
        "CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].[Rh+3].[Rh+3]"
    ] = "CC(=O)[O-].[Rh+2]"
    molecule_replacements[
        "[CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].[Rh+3]]"
    ] = "CC(=O)[O-].[Rh+2]"
    molecule_replacements[
        "[CC(C)(C)[P]([Pd][P](C(C)(C)C)(C(C)(C)C)C(C)(C)C)(C(C)(C)C)C(C)(C)C]"
    ] = "CC(C)(C)[PH]([Pd][PH](C(C)(C)C)(C(C)(C)C)C(C)(C)C)(C(C)(C)C)C(C)(C)C"
    molecule_replacements[
        "CCCC[N+](CCCC)(CCCC)CCCC.CCCC[N+](CCCC)(CCCC)CCCC.CCCC[N+](CCCC)(CCCC)CCCC.[Br-].[Br-].[Br-]"
    ] = "CCCC[N+](CCCC)(CCCC)CCCC.[Br-]"
    molecule_replacements["[CCO.CCO.CCO.CCO.[Ti]]"] = "CCO[Ti](OCC)(OCC)OCC"
    molecule_replacements[
        "[CC[O-].CC[O-].CC[O-].CC[O-].[Ti+4]]"
    ] = "CCO[Ti](OCC)(OCC)OCC"
    molecule_replacements[
        "[Cl[Ni]Cl.c1ccc(P(CCCP(c2ccccc2)c2ccccc2)c2ccccc2)cc1]"
    ] = "Cl[Ni]1(Cl)[P](c2ccccc2)(c2ccccc2)CCC[P]1(c1ccccc1)c1ccccc1"
    molecule_replacements[
        "[Cl[Pd](Cl)([P](c1ccccc1)(c1ccccc1)c1ccccc1)[P](c1ccccc1)(c1ccccc1)c1ccccc1]"
    ] = "Cl[Pd](Cl)([PH](c1ccccc1)(c1ccccc1)c1ccccc1)[PH](c1ccccc1)(c1ccccc1)c1ccccc1"
    molecule_replacements["[Cl[Pd+2](Cl)(Cl)Cl.[Na+].[Na+]]"] = "Cl[Pd]Cl"
    molecule_replacements["Karstedt catalyst"] = "C[Si](C)(C=C)O[Si](C)(C)C=C.[Pt]"
    molecule_replacements["Karstedt's catalyst"] = "C[Si](C)(C=C)O[Si](C)(C)C=C.[Pt]"
    molecule_replacements["[O=C([O-])[O-].[Ag+2]]"] = "O=C([O-])[O-].[Ag+]"
    molecule_replacements["[O=S(=O)([O-])[O-].[Ag+2]]"] = "O=S(=O)([O-])[O-].[Ag+]"
    molecule_replacements["[O=[Ag-]]"] = "O=[Ag]"
    molecule_replacements["[O=[Cu-]]"] = "O=[Cu]"
    molecule_replacements["[Pd on-carbon]"] = "[C].[Pd]"
    molecule_replacements["[TEA]"] = "OCCN(CCO)CCO"
    molecule_replacements["[Ti-superoxide]"] = "O=[O-].[Ti]"
    molecule_replacements[
        "[[Pd].c1ccc(P(c2ccccc2)c2ccccc2)cc1]"
    ] = "[Pd].c1ccc(P(c2ccccc2)c2ccccc2)cc1"
    molecule_replacements[
        "[c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd-4]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1]"
    ] = "c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1"
    molecule_replacements[
        "[c1ccc([P]([Pd][P](c2ccccc2)(c2ccccc2)c2ccccc2)(c2ccccc2)c2ccccc2)cc1]"
    ] = "c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1"
    molecule_replacements[
        "[c1ccc([P](c2ccccc2)(c2ccccc2)[Pd]([P](c2ccccc2)(c2ccccc2)c2ccccc2)([P](c2ccccc2)(c2ccccc2)c2ccccc2)[P](c2ccccc2)(c2ccccc2)c2ccccc2)cc1]"
    ] = "c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1"
    molecule_replacements["[sulfated tin oxide]"] = "O=S(O[Sn])(O[Sn])O[Sn]"
    molecule_replacements[
        "[tereakis(triphenylphosphine)palladium(0)]"
    ] = "c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1"
    molecule_replacements[
        "tetrakistriphenylphosphine palladium"
    ] = "c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1"
    molecule_replacements["[zeolite]"] = "O=[Al]O[Al]=O.O=[Si]=O"

    # Molecules found among the most common names in molecule_names
    molecule_replacements["TEA"] = "OCCN(CCO)CCO"
    molecule_replacements["hexanes"] = "CCCCCC"
    molecule_replacements["Hexanes"] = "CCCCCC"
    molecule_replacements["hexanes ethyl acetate"] = "CCCCCC.CCOC(=O)C"
    molecule_replacements["EtOAc hexanes"] = "CCCCCC.CCOC(=O)C"
    molecule_replacements["EtOAc-hexanes"] = "CCCCCC.CCOC(=O)C"
    molecule_replacements["ethyl acetate hexanes"] = "CCCCCC.CCOC(=O)C"
    molecule_replacements["cuprous iodide"] = "[Cu]I"
    molecule_replacements["N,N-dimethylaminopyridine"] = "n1ccc(N(C)C)cc1"
    molecule_replacements["dimethyl acetal"] = "CN(C)C(OC)OC"
    molecule_replacements["cuprous chloride"] = "Cl[Cu]"
    molecule_replacements["N,N'-carbonyldiimidazole"] = "O=C(n1cncc1)n2ccnc2"
    # SiO2
    # Went down the list of molecule_names until frequency was 806

    # Iterate over the dictionary and canonicalize each SMILES string
    for key, value in molecule_replacements.items():
        mol = Chem.MolFromSmiles(value)

        if mol is not None:
            molecule_replacements[key] = Chem.MolToSmiles(mol)
    return molecule_replacements


# %%
molecule_replacements = build_replacements()

# %%
values = list(molecule_replacements.values())
values = values[17:18]
for value in values:
    mol = Chem.MolFromSmiles(value)
    print(value)

# %%
agents = ["carbon", "Al", "Pd", "c", "Al"]
metals = [
    "Li",
    "Be",
    "Na",
    "Mg",
    "Al",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
]
agents = [agent for agent in agents if any(metal in agent for metal in metals)] + [
    agent for agent in agents if not any(metal in agent for metal in metals)
]

# %%
agents

# %% [markdown]
# # Check the overlap between two solvents lists

# %%
import pandas as pd

# %%
ucb_solvents = pd.read_csv("data/ucb_solvents.csv")
solvents = pd.read_csv("data/solvents.csv")

cas1 = ucb_solvents["cas_number"].tolist()
cas2 = solvents["cas_number"].tolist()

# %%
cas1[-1]

# %%
count = -1
for cas in cas1:
    count += 1
    if cas not in cas2:
        print(ucb_solvents["solvent_name"][count])

# %%
solvents.loc[solvents["cas_number"] == "7732-18-5"]

# %% [markdown]
# # Docs

# %%
# USPTO Cleaning docs

"""
After running USPTO_extraction.py, this script will merge and apply further cleaning to the data.

Example: python USPTO_cleaning.py --clean_data_file_name=USPTO_data.csv --consistent_yield=True --num_reactant=1 --num_product=1 --num_solv=1 --num_agent=1 --num_cat=1 --num_reag=1 --min_frequency_of_occurrence=100

    Args:
1) clean_data_file_name, consistent_yield, num_reactant, num_product, num_solv, num_agent, num_cat, num_reag, min_frequency_of_occurrence

    
    
    
    
    
"""

# %%
"""
After running USPTO_extraction.py, this script will merge and apply further cleaning to the data.

    Example: 

python USPTO_cleaning.py --clean_data_file_name=cleaned_USPTO --consistent_yield=True --num_reactant=5 --num_product=5 --num_solv=2 --num_agent=3 --num_cat=0 --num_reag=0 --min_frequency_of_occurrence=100

    Args:
    
1) clean_data_file_name: (str) The filepath where the cleaned data will be saved
2) consistent_yield: (bool) Remove reactions with inconsistent reported yields (e.g. if the sum is under 0% or above 100%. Reactions with nan yields are not removed) 
3) - 8) num_reactant, num_product, num_solv, num_agent, num_cat, num_reag: (int) The number of molecules of that type to keep. Keep in mind that if merge_conditions=True in USPTO_extraction, there will only be agents, but no catalysts/reagents, and if merge_conditions=False, there will only be catalysts and reagents, but no agents. Agents should be seen as a 'parent' category of reagents and catalysts; solvents should fall under this category as well, but since the space of solvents is more well defined (and we have a list of the most industrially relevant solvents which we can refer to), we can separate out the solvents. Therefore, if merge_conditions=True, num_catalyst and num_reagent should be set to 0, and if merge_conditions=False, num_agent should be set to 0. It is recommended to set merge_conditions=True, as we don't believe that the original labelling of catalysts and reagents that reliable; furthermore, what constitutes a catalyst and what constitutes a reagent is not always clear, adding further ambiguity to the labelling, so it's probably best to merge these.
9) min_frequency_of_occurrence: (int) The minimum number of times a molecule must appear in the dataset to be kept. Infrequently occuring molecules will probably add more noise than signal to the dataset, so it is best to remove them.

    Functionality:

1) Merge the pickle files from USPTO_extraction.py into a df
2) Remove reactions with too many reactants, products, sovlents, agents, catalysts, and reagents (num_reactant, num_product, num_solv, num_agent, num_cat, num_reag)
3) Remove reactions with inconsistent yields (consistent_yield)
4) Remove molecules that appear less than min_frequency_of_occurrence times
5) Remove reactions that have a molecule represented by an unresolvable name. This is often an english name or a number.
6) Remove duplicate reactions
7) Pickle the final df

    Output:

1) A pickle file containing the cleaned data
"""

# %% [markdown]
# # Inspect rows with weird names

# %%
import pickle
import pandas as pd
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

# %%
# import cleaned_USPTO.pkl
df = pd.read_pickle("data/USPTO/cleaned_USPTO.pkl")


# %%
def merge_extracted_ords():
    # create one big df of all the pickled data
    folder_path = "data/USPTO/extracted_ords/"
    onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    full_df = pd.DataFrame()
    for file in tqdm(onlyfiles):
        if file[0] != ".":  # We don't want to try to unpickle .DS_Store
            filepath = folder_path + file
            unpickled_df = pd.read_pickle(filepath)
            full_df = pd.concat([full_df, unpickled_df], ignore_index=True)

    return full_df


# %%
df = merge_extracted_ords()

# %%
smaller_df = df.iloc[:1000].copy()

# %%
# Assuming your dataframe is named df
solution_rows = smaller_df[
    smaller_df.apply(lambda row: row.astype(str).str.contains("solution").any(), axis=1)
].head(5)

# Print the first 5 rows that contain 'solution'
solution_rows

# %%
smaller_df.columns

# %%
smaller_df.iloc[246].dropna()

# %%
smaller_df.iloc[198]["mapped_rxn_0"]

# %%
count = -1
for x in df["agent_0"]:
    count += 1
    if x != None and len(x) > 0:
        if x.isdigit():
            print(x)
            print(count)
            break

# %%
df["solvent_2"].dropna()

# %%
len(df["agent_3"].dropna())

# %% [markdown]
# # Inspect the clean data

# %%
import pandas as pd

# %%
# import cleaned_USPTO.pkl
df = pd.read_pickle("data/USPTO/cleaned_USPTO.pkl")

# %%
len(set(list(df["solvent_0"].dropna())))

# %%
import matplotlib.pyplot as plt

# Count the frequency of each unique value
counts = pd.Series.value_counts(df["agent_0"].dropna())

# Create a list of bins for the histogram
bins = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
]

# Plot the histogram
plt.hist(counts, bins=bins)

# Set the x-axis and y-axis labels
plt.xlabel("Frequency Count")
plt.ylabel("Number of Items")

# Show the plot
plt.show()


# %%
counts[-100000:]

# %%
type(counts)

# Count the frequency of each unique value
counts = pd.Series.value_counts(counts)

# Get the count of the value 1
count_of_ones = counts[1]

# Print the count of the value 1
print(count_of_ones)


# %%
df["temperature_0"].dropna()

# %%
# a,b,c = 1,2,3
# assert a<c b<c, "error"

# %%

import pandas as pd

a = [None,"ased","as",""]
b = [1,None,3,4]

p,q = list(zip(*[(x,j) for x,j in zip(a,b) if (x not in ["", None]) and (not pd.isna(x))]))

# %%

import pandas as pd

df = pd.DataFrame(
    {
        "yield_000": [
            "a",
            "a",
            "a",
            "a",
        ],
        "yield_001": [
            "b",
            "b",
            None,
            "b",
        ],
        "yield_002": [
            None,
            None,
            None,
            "c",
        ],
        "reactant_001": [
            "b",
            "b",
            "b",
            None,
        ],
        "reactant_000": [
            None,
            None,
            None,
            "a",
        ],
        "product_000": [
            "a",
            None,
            "a",
            None,
        ],
        "product_001": [
            None,
            "b",
            None,
            None,
        ],
        "product_002": [
            "c",
            None,
            "c",
            "c",
        ],
    }
)
# %%

molecule_type = "reactant"

from orderly.clean.cleaner import Cleaner

_product = Cleaner._get_columns_beginning_with_str(
    columns=df.columns,
    target_strings=("product",),
)

_yield = Cleaner._get_columns_beginning_with_str(
    columns=df.columns,
    target_strings=("yield",),
)

ordering_target_columns = sorted(ordering_target_columns)

print(ordering_target_columns)

def sort_row_old(row):
    return pd.Series(
        sorted(row, key=lambda x: pd.isna(x)), index=row.index
    )

def sort_row(row):
    return pd.Series(
        sorted(row, key=lambda x: pd.isna(x)), index=row.index
    )

def sort_row(row):
    idx = row.index
    row = row.sort_values(na_position="last")
    row.index = idx
    return row

print(
    df.loc[
        :, ordering_target_columns
    ].apply(
        sort_row, axis=1
    )
)

print(
df.loc[
    :, ordering_target_columns
].apply(
    lambda x: sort_row(x), axis=1
)
)
# %%

def sort_row_relative(row, to_sort, to_keep_ordered):

    target_row = row[to_sort].reset_index(drop=True).sort_values(na_position="last")
    rel_row = row[to_keep_ordered].reset_index(drop=True)

    rel_row = rel_row[target_row.index]
    rel_row.index = to_keep_ordered
    target_row.index = to_sort

    row = pd.concat([target_row, rel_row])
    
    return row

print("Before")
print(
    df.loc[
        :, _product+_yield
    ]
)
print("After")
print(
    df.loc[
        :, _product+_yield
    ].apply(
        lambda x: sort_row_relative(x, _product, _yield), axis=1
    )
)

# %%

%%timeit

df.loc[
        :, _product+_yield
    ].apply(
        lambda x: sort_row_relative(x, _product, _yield), axis=1
    )

# %%

%%timeit

df.loc[
    :, ordering_target_columns
].apply(
    sort_row, axis=1
)
# %%
%%timeit
df.loc[
    :, ordering_target_columns
].apply(
    lambda x: sort_row(x), axis=1
)
# %%
%%timeit

df.loc[
    :, ordering_target_columns
].apply(
    sort_row_old, axis=1
)
# %%
import pandas as pd

pd.DataFrame(
    {
        "product_000": [
            "a",
            "a",
            "a",
            None,
        ],
        "product_001": [
            "b",
            "b",
            "b",
            None,
        ],
        "product_002": [
            None,
            "c",
            None,
            None,
        ],
        "yield_000": [
            "a",
            "a",
            "a",
            None,
        ],
        "yield_001": [
            "b",
            "b",
            "b",
            None,
        ],
        "yield_002": [
            None,
            "b",
            "c",
            None,
        ],
    }
)
# %%

pd.DataFrame(
    {
        "product_000": [
            "a",
            "a",
            None,
            None,
        ],
        "product_001": [
            None,
            "b",
            "a",
            None,
        ],
        "product_002": [
            "b",
            "c",
            "b",
            None,
        ],
        "yield_000": [
            "a",
            "a",
            "c",
            None,
        ],
        "yield_001": [
            None,
            "b",
            "a",
            None,
        ],
        "yield_002": [
            "b",
            "c",
            "b",
            None,
        ],
    }
)
# %%
