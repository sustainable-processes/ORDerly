# %% [markdown]
# # Check that solvents.csv is well set up

# %%
import pandas as pd
from rdkit import Chem

# %%
solvents_df = pd.read_csv('orderly/data/solvents.csv')
solvents_df.columns
solvents_smiles = solvents_df['smiles']

# %%
solvents_df.columns


# %%
# check all smiles are canonicalisable
# Canonicalise and remove stoichiometry
def clean_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(smiles)
        return smiles
    else:
        return Chem.MolToSmiles(mol, isomericSmiles=False)




# %%
clean_smiles('C[Pd]')

# %%
# Apply the function to all columns in the DataFrame
solvents_smiles.apply(clean_smiles)


# %%
#check for duplicates
strings_lst = solvents_df[['solvent_name_1', 'solvent_name_2', 'solvent_name_3', 'canon_smiles']].values.flatten().tolist()
strings_lst = [x for x in strings_lst if not pd.isna(x)]

# manually remove duplicates to ensure we don't lose any data
# create dictionary to count occurrences
count_dict = {}
for item in strings_lst:
    count_dict[item] = count_dict.get(item, 0) + 1

# extract elements with count of 2 or more
duplicates = [item for item, count in count_dict.items() if count >= 2]

# %%


# %%
duplicates

# %% [markdown]
# # Save SMILES column with canon smiles

# %%
import pandas as pd
from rdkit import Chem

solvents_df = pd.read_csv('orderly/data/solvents.csv')
solvents_df.columns
solvents_smiles = solvents_df['smiles']

# Canonicalise and remove stoichiometry
def clean_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=False)

# Apply the function to all columns in the DataFrame
canon_smiles = solvents_smiles.apply(clean_smiles)




# %%
solvents_df['canon_smiles'] = canon_smiles

# %%
solvents_df.to_csv('orderly/data/solvents_2.csv', index=False)

# %% [markdown]
# # Fixing solvents csv

# %%
a = 'None, Cc1ccc(S(=O)(=O)[O-])cc1, [H-], c1ccc([P](c2ccccc2)(c2ccccc2)[Pd]([P](c2ccccc2)(c2ccccc2)c2ccccc2)([P](c2ccccc2)(c2ccccc2)c2ccccc2)[P](c2ccccc2)(c2ccccc2)c2ccccc2)cc1, [H-], Cl, None, c1ccc([P](c2ccccc2)(c2ccccc2)[Pd]([P](c2ccccc2)(c2ccccc2)c2ccccc2)([P](c2ccccc2)(c2ccccc2)c2ccccc2)[P](c2ccccc2)(c2ccccc2)c2ccccc2)cc1, isoparaffin hydrocarbons, None, O=C(O)CCC(=O)O, None, c1c[nH]cn1, O=C(O)Cc1ccc(Br)cc1, O=C([O-])[O-], None, None, [Al+3], [Al+3], [Al+3], None, Cl, Cc1ccc(S(=O)(=O)O)cc1, crude product, [F-], [Ni], None, None, None, [F-], [Al+3], [Al+3], None, [Al+3], CCCCCC, COc1cc(OC)nc(NC(=O)NS(=O)(=O)Nc2ccccc2C(O)C2CC2)n1, Cl, Cl, O, None, None, O=C([O-])[O-], O=C([O-])[O-], Cl, [Na+], O=C([O-])[O-], [Pd], Cl, None, Cl, O, [K+], None, Cl, Cl, Cl, Cl, [Pd], None, None, Cl, Cl, C(=NC1CCCCC1)=NC1CCCCC1, C[C@H]1CC[C@@H]2[C@@H](C1)OC(=O)[C@@H]2C, CN(C)c1ccccn1, None, IR(KBr), O=S(=O)(O)O, CCOC(=O)Cl, C[Si](C)(C)OS(=O)(=O)C(F)(F)F, CCOC(=O)N=NC(=O)OCC, None, Br, CCOC(=O)N=NC(=O)OCC, Cc1ccc(S(=O)(=O)O)cc1, 1h, [H-], 1h, [H-], C12H18N4O5S3, Cl, [Pd+2], CCOC(=O)N=NC(=O)OCC, [H-], [Li+], Cl, None, None, [K+], [Li+], None, None, [K+], [Li+], None, Cl, Cl, None, None, [Pd]'
b = 'None, Cc1ccc(S(=O)(=O)[O-])cc1, [H-], c1ccc([P](c2ccccc2)(c2ccccc2)[Pd]([P](c2ccccc2)(c2ccccc2)c2ccccc2)([P](c2ccccc2)(c2ccccc2)c2ccccc2)[P](c2ccccc2)(c2ccccc2)c2ccccc2)cc1, [H-], Cl, None, c1ccc([P](c2ccccc2)(c2ccccc2)[Pd]([P](c2ccccc2)(c2ccccc2)c2ccccc2)([P](c2ccccc2)(c2ccccc2)c2ccccc2)[P](c2ccccc2)(c2ccccc2)c2ccccc2)cc1, isoparaffin hydrocarbons, None, O=C(O)CCC(=O)O, None, c1c[nH]cn1, O=C(O)Cc1ccc(Br)cc1, O=C([O-])[O-], None, None, [Al+3], [Al+3], [Al+3], None, Cl, Cc1ccc(S(=O)(=O)O)cc1, crude product, [F-], [Ni], None, None, None, [F-], [Al+3], [Al+3], None, [Al+3], CCCCCC, COc1cc(OC)nc(NC(=O)NS(=O)(=O)Nc2ccccc2C(O)C2CC2)n1, Cl, Cl, O, None, None, O=C([O-])[O-], O=C([O-])[O-], Cl, [Na+], O=C([O-])[O-], [Pd], Cl, None, Cl, O, [K+], None, Cl, Cl, Cl, Cl, [Pd], None, None, Cl, Cl, C(=NC1CCCCC1)=NC1CCCCC1, C[C@H]1CC[C@@H]2[C@@H](C1)OC(=O)[C@@H]2C, CN(C)c1ccccn1, C1CCC2=NCCCN2CC1, IR(KBr), O=S(=O)(O)O, CCOC(=O)Cl, C[Si](C)(C)OS(=O)(=O)C(F)(F)F, CCOC(=O)N=NC(=O)OCC, None, Br, CCOC(=O)N=NC(=O)OCC, Cc1ccc(S(=O)(=O)O)cc1, 1h, [H-], 1h, [H-], C12H18N4O5S3, Cl, [Pd+2], CCOC(=O)N=NC(=O)OCC, [H-], [Li+], Cl, None, None, [K+], [Li+], None, None, [K+], [Li+], None, Cl, CCN(C(C)C)C(C)C, None, None, [Pd]'


# %%
a.split(',')[:7]

# %%
b.split(',')[:7]

# %%
a ==b


# %%
import pandas as pd
from rdkit import Chem
import numpy as np

# %%
solvents_df = pd.read_csv('orderly/data/solvents.csv')

# %%
# create set of strings in df1
strings_to_remove = solvents_df.values.flatten().tolist()
strings_to_remove = [x for x in strings_to_remove if not pd.isna(x)]

# %%
len(set(strings_to_remove))

# %%
len(strings_to_remove)

# %%
# manually remove duplicates to ensure we don't lose any data
# create dictionary to count occurrences
count_dict = {}
for item in strings_to_remove:
    count_dict[item] = count_dict.get(item, 0) + 1

# extract elements with count of 2 or more
duplicates = [item for item, count in count_dict.items() if count >= 2]

# %%
duplicates

# %%
df2 = df2[~df2['solvent_name_1'].isin(strings_to_remove)]
df2 = df2[~df2['solvent_name_2'].isin(strings_to_remove)]
df2 = df2[~df2['solvent_name_3'].isin(strings_to_remove)]

# %%
df2.to_csv('orderly/data/subset_2_.csv', index=False)

# %% [markdown]
# # Check for duplicates in the three name columns

# %%
# we want to look for duplicates in the 'solvent_name' columns
solvent_names = solvents_df[['solvent_name_1', 'solvent_name_2', 'solvent_name_3']]
# check for duplicates
duplicates = solvent_names.duplicated()

# print duplicate rows
print(solvent_names[duplicates])

# %%


# %%
solvent_names_list = solvent_names.values.flatten().tolist()
solvent_names_list = [x for x in solvent_names_list if not pd.isna(x)]


# %%
print(len(solvent_names_list))

# %%
print(len(set(solvent_names_list)))

# %%
# manually remove duplicates to ensure we don't lose any data
# create dictionary to count occurrences
count_dict = {}
for item in solvent_names_list:
    count_dict[item] = count_dict.get(item, 0) + 1

# extract elements with count of 2 or more
duplicates = [item for item, count in count_dict.items() if count >= 2]

# %%
len(duplicates)

# %%
duplicates

# %%


# %%
# check for any trailing or leading spaces
for s in solvent_names_list:
    if s != s.strip():
        print(s)

# %%
# # make lower
# # make specified columns lower case
# solvents_df[['solvent_name_1', 'solvent_name_2', 'solvent_name_3']] = solvents_df[['solvent_name_1', 'solvent_name_2', 'solvent_name_3']].applymap(lambda x: x.lower() if type(x) == str else x)

# # save to csv
# solvents_df.to_csv('orderly/data/solvents.csv', index=False)

# %% [markdown]
# # Use pura on each column

# %%
import pandas as pd

# Import pura
from pura.resolvers import resolve_identifiers
from pura.compound import CompoundIdentifierType
from pura.services import PubChem, CIR, Opsin, CAS, ChemSpider, STOUT

solvents_df = pd.read_csv('orderly/data/solvents.csv')

# %%
def apply_pura_name_to_smiles(lst, services=[PubChem(autocomplete=True), Opsin(), CIR(),]):
    resolved = resolve_identifiers(
        lst,
        input_identifer_type=CompoundIdentifierType.NAME,
        output_identifier_type=CompoundIdentifierType.SMILES,
        services=services,
        agreement=2,
        silent=True,
    )
    return resolved

def apply_pura_cas_to_smiles(lst):
    resolved = resolve_identifiers(
        lst,
        input_identifer_type=CompoundIdentifierType.CAS_NUMBER,
        output_identifier_type=CompoundIdentifierType.SMILES,
        services=[CAS()],
        agreement=1,
        silent=True,
    )
    return resolved

# %%
col1_lst_1 = solvents_df['solvent_name_1'].dropna().tolist()
col1_lst_2 = solvents_df['solvent_name_2'].dropna().tolist()
col1_lst_3 = solvents_df['solvent_name_3'].dropna().tolist()
names = col1_lst_1 + col1_lst_2 + col1_lst_3

# %%
len(set(names)) == len(names)
print(len(names))

# %%
names_pura = apply_pura_name_to_smiles(names)

# %%
names_dict = dict(names_pura)

# %%
names_dict

# %%
solvents_df.columns

# %%
cas_names = apply_pura_cas_to_smiles(solvents_df['cas_number'].dropna().tolist())

# %%
cas_names_dict = dict(cas_names)

# %%
cas_names_dict_2 ={key: value[0] if value else '' for key, value in cas_names_dict.items()}

# %%
names_dict_2 = {key: value[0] if value else '' for key, value in names_dict.items()}


# %%
replacement_df = solvents_df.replace(names_dict_2)

# %%
replacement_df_2 = replacement_df.replace(cas_names_dict_2)

# %%
replacement_df_2.to_csv('orderly/data/pura_solvents_.csv', index=False)

# %% [markdown]
# # Resolve pura smiles

# %%
pura_solvents = pd.read_csv('orderly/data/pura_solvents.csv')

# %%
pura_solvents

# %%
# Canonicalise and remove stoichiometry
def clean_smiles(smiles):
    if pd.isna(smiles):
        return smiles
    else:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, isomericSmiles=False)

# Apply the function to all columns in the DataFrame
df = pura_solvents.applymap(clean_smiles)


# %%
df = df.drop('smiles_5', axis=1)

# %%
df = df.drop('final_smiles', axis=1)

# %%
df

# %%
def get_final_smiles(row):
    row_dropna = row.dropna()
    smiles_set = set(row_dropna)
    if len(smiles_set) == 1 and len(row_dropna) >= 1:
        return smiles_set.pop()
    else:
        return np.nan

# Apply the function to each row in the DataFrame to generate the 'final_smiles' column
df['final_smiles'] = df.apply(get_final_smiles, axis=1)

# %%
df['final_smiles'].dropna()

# %%
# need to manually reosolve 22 compounds where there's disagreement
df.to_csv('orderly/data/solvents_check_agreement.csv', index=False)

# %%



