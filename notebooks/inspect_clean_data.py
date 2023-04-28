# %% [markdown]
# # Inspect extracted data (USPTO, trust = False)

# %%
import pandas as pd
from orderly.extract.canonicalise import remove_mapping_info_and_canonicalise_smiles

# %%
remove_mapping_info_and_canonicalise_smiles("O=[Si]=O.O=[Al]O[Al]=O")

# %%
remove_mapping_info_and_canonicalise_smiles("[Ti]")

# %%
d = get_molecule_replacements()

# %%
for val in d.values():
    canon = remove_mapping_info_and_canonicalise_smiles(val)
    if canon == None:
        print(val)

# %%
remove_mapping_info_and_canonicalise_smiles("CC(OCC)OCC")

# %%
import pandas as pd
import numpy as np

# %%
# create a sample dataframe with NaN, NaT and None values
df = pd.DataFrame(
    {"A": [None, None, None], "B": [np.nan, pd.NaT, None], "C": [np.nan, pd.NaT, None]}
)

# assert that the dataframe only contains NaN or NaT values
assert df.isnull().all().all() == True

# %%
assert df.isnull().all().all() == True

# %%
df

# %%


# %%


# %%
if None is None:
    print("hi")

# %%
"ice" in ["ice", "ice water"]

# %%
mole_id_list = [None, "a", "b", None, "c", ""]
mole_id_list_without_none = [x for x in mole_id_list if x not in ["", None]]

# %%
mole_id_list_without_none

# %%
path = "/Users/dsw46/Library/CloudStorage/OneDrive-UniversityofCambridge/Datasets/orderly/orderly_data/18_4/uspto_no_trust/extracted_ords/uspto-grants-1976_01.parquet"
df = pd.read_parquet(path)

# %%
df.shape[0]

# %%


# %%
i = 412
print(df.iloc[i]["procedure_details"])
print(df.iloc[i]["rxn_str"])


# %%
print(df.iloc[i]["procedure_details"])

# %%
remove_mapping_info_and_canonicalise_smiles("C[Si](C)(C=C)O[Si](C)(C)C=C[Pt]")

# %%
from rdkit import Chem

mol = Chem.MolFromSmiles("Cl")
smiles = Chem.MolToSmiles(mol)
smiles


# %%
"Br[CH2:2][C:3]1[CH:4]=[CH:5][C:6]2[O:15][C:10]3=[N:11][CH:12]=[CH:13][CH:14]=[C:9]3[C:8](=[O:16])[C:7]=2[CH:17]=1.[CH3:18][N:19](C)C=O.[C-]#N.[Na+]>O>[C:18]([CH2:2][C:3]1[CH:4]=[CH:5][C:6]2[O:15][C:10]3=[N:11][CH:12]=[CH:13][CH:14]=[C:9]3[C:8](=[O:16])[C:7]=2[CH:17]=1)#[N:19]"

# %%
"Br[CH2:2][C:3]1[CH:4]=[CH:5][C:6]2[O:15][C:10]3=[N:11][CH:12]=[CH:13][CH:14]=[C:9]3[C:8](=[O:16])[C:7]=2[CH:17]=1.[CH3:18][N:19](C)C=O.[C-]#N.[Na+]".split(
    "."
)

# %%
df.iloc[i]

# %% [markdown]
# # Inspect extracted data (trust = True)

# %%
# path = '/Users/dsw46/Library/CloudStorage/OneDrive-UniversityofCambridge/Datasets/orderly/orderly_data/18_4/uspto_trust/extracted_ords/uspto-grants-1976_01.parquet'
# df = pd.read_parquet(path)

# %% [markdown]
# # Inspect test data

# %%


# %%
import pandas as pd
import numpy as np

df = pd.read_parquet(
    "orderly/data/test_data/extracted_ord_test_data_dont_trust_labelling/extracted_ords/uspto-grants-1978_12.parquet"
)

# %%
df["reactant_005"].replace({None: np.nan})

# %%
False == 0

# %%
False == 0

# %%
# 0 not False

# %%
df["date_of_experiment"].dropna()

# %%
test_df = pd.DataFrame({"a": [None, 1, 2, 3], "b": [1, 2, 3, 4]})

# %%
test_df.replace({None: np.nan})

# %% [markdown]
# # Inspect clean data (USPTO, trust = False)

# %%
path = "/Users/dsw46/Library/CloudStorage/OneDrive-UniversityofCambridge/Datasets/orderly/orderly_data/uspto_no_trust/orderly_ord.parquet"
df = pd.read_parquet(path)

# %%
df.shape

# %%
df.iloc[10]  # ['rxn_str_0']

# %%
"[S:1]([O-:5])([O-:4])(=[O:3])=[O:2].[NH4+:6].[NH4+]>O>[S:1](=[O:3])(=[O:2])([OH:5])[O-:4].[NH4+:6].[S:1]([O-:5])([O-:4])(=[O:3])=[O:2].[NH4+:6].[NH4+:6]"

# %%
"[Br:1][CH2:2][CH2:3][OH:4].[CH2:5]([S:7](Cl)(=[O:9])=[O:8])[CH3:6].CCOCC>C(N(CC)CC)C>[CH2:5]([S:7]([O:4][CH2:3][CH2:2][Br:1])(=[O:9])=[O:8])[CH3:6]"

# %%
a, b, c = "a>>b".split(">")

# %%
a.split(".")

# %%

# %% [markdown]
# # Open rxn object

# %%


# %%
from ord_schema import message_helpers, validations
from ord_schema.proto import dataset_pb2

# %%
# Load Dataset message
# pb = 'data/ord//02/ord_dataset-02ee2261663048188cf6d85d2cc96e3f.pb.gz'
pb = "data/ord//a0/ord_dataset-a0eff6fe4b4143f284f0fc5ac503acad.pb.gz"
# pb = 'orderly/data/ord_test_data/00/ord_dataset-00005539a1e04c809a9a78647bea649c.pb.gz'
pb = "data/ord//0b/ord_dataset-0bb2e99daa66408fb8dbd6a0781d241c.pb.gz"
data = message_helpers.load_message(pb, dataset_pb2.Dataset)

# %%
data.reactions[0]

# %%
data.reactions[10].identifiers[0].value

# %%
"CC1N=CC2C(C=1)=C([N+]([O-])=O)C=CC=2.[Cl:15][C:16]1[CH:25]=[CH:24][C:23]([N+:26]([O-:28])=[O:27])=[C:22]2[C:17]=1[CH:18]=[CH:19][N:20]=[CH:21]2.Cl.CC1N=CC2C(C=1)=C([N+]([O-])=O)C=CC=2.[IH:44]>>[IH:44].[Cl:15][C:16]1[CH:25]=[CH:24][C:23]([N+:26]([O-:28])=[O:27])=[C:22]2[C:17]=1[CH:18]=[CH:19][N:20]=[CH:21]2"

# %%
"CC1N=CC2C(C=1)=C([N+]([O-])=O)C=CC=2.[Cl:15][C:16]1[CH:25]=[CH:24][C:23]([N+:26]([O-:28])=[O:27])=[C:22]2[C:17]=1[CH:18]=[CH:19][N:20]=[CH:21]2.Cl.CC1N=CC2C(C=1)=C([N+]([O-])=O)C=CC=2.[IH:44]".split(
    "."
)

# %%
"[IH:44].[Cl:15][C:16]1[CH:25]=[CH:24][C:23]([N+:26]([O-:28])=[O:27])=[C:22]2[C:17]=1[CH:18]=[CH:19][N:20]=[CH:21]2".split(
    "."
)

# %%
remove_mapping_info_and_canonicalise_smiles("[Na].[OH-]")

# %%


# %%
sorted(["I", "O=[N+]([O-])c1ccc(Cl)c2ccncc12", "Cc1cc2c([N+](=O)[O-])cccc2cn1", "Cl"])

# %%
dict = {"a": 5, "b": 10}
s = pd.Series(dict)

# %%
"a" in s.keys()

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%
"[CH2:1]([CH:10]1[C:18]2[C:13](=[CH:14][CH:15]=[CH:16][CH:17]=2)[CH2:12][C:11]1=O)[C:2]([C:4]1[CH:9]=[CH:8][CH:7]=[CH:6][CH:5]=1)=O.[NH2:20][C:21]1[CH:29]=[C:25]([C:26]([OH:28])=[O:27])[C:24]([OH:30])=[CH:23][CH:22]=1>C(O)(=O)C>[C:26]([C:25]1[CH:29]=[C:21]([N:20]2[C:2]([C:4]3[CH:9]=[CH:8][CH:7]=[CH:6][CH:5]=3)=[CH:1][C:10]3[C:18]4[CH:17]=[CH:16][CH:15]=[CH:14][C:13]=4[CH2:12][C:11]2=3)[CH:22]=[CH:23][C:24]=1[OH:30])([OH:28])=[O:27]"

# %%
"[CH:1]1[C:14]2[C:13](=[O:15])[C:12]3[C:7](=[CH:8][CH:9]=[CH:10][CH:11]=3)[C:6](=[O:16])[C:5]=2[CH:4]=[CH:3][CH:2]=1.[N+:17]([O-])([OH:19])=[O:18]>>[N+:17]([C:8]1[C:7]2[C:6](=[O:16])[C:5]3[C:14](=[CH:1][CH:2]=[CH:3][CH:4]=3)[C:13](=[O:15])[C:12]=2[CH:11]=[CH:10][CH:9]=1)([O-:19])=[O:18]"

# %%
query = "[CH3:1][C:2]1[NH:3][C:4]2[C:9]([CH:10]=1)=[CH:8][CH:7]=[CH:6][CH:5]=2.[Mg].C(I)C.CC.Cl[CH2:18][CH2:19][C:20]([N:22]1[CH2:27][CH2:26][N:25]([C:28]2[CH:33]=[CH:32][CH:31]=[CH:30][CH:29]=2)[CH2:24][CH2:23]1)=[O:21]>CCOCC.C(O)(=O)C.C1C=CC=CC=1>[CH3:1][C:2]1[NH:3][C:4]2[C:9]([C:10]=1[CH2:18][CH2:19][C:20]([N:22]1[CH2:27][CH2:26][N:25]([C:28]3[CH:33]=[CH:32][CH:31]=[CH:30][CH:29]=3)[CH2:24][CH2:23]1)=[O:21])=[CH:8][CH:7]=[CH:6][CH:5]=2"
for i, rxn in enumerate(data.reactions):
    if rxn.identifiers[0].value == query:
        print(i)
        break


# %%
data.reactions[i]

# %% [markdown]
# ##  ψ-acid chloride

# %%
path = "/Users/dsw46/Library/CloudStorage/OneDrive-UniversityofCambridge/Datasets/orderly/orderly_data/18_4/uspto_no_trust/extracted_ords/uspto-grants-1976_01.parquet"
df = pd.read_parquet(path)
i = 412

# %%
