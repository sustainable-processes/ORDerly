# %% [markdown]
# # Read in data

# %%
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import numpy as np
from tqdm import tqdm
from tqdm import trange
import matplotlib.pyplot as plt
# import pyarrow as pa

# %%
"""
Disables RDKit whiny logging.
"""
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog('rdApp.error')

# %%
# This is the data preprocessed in USPTO_preprocessing.ipynb
# There's around 500k reactions, and columns for reactant, product, solvent, reagent, etc.
# So there's quite a bit more data than in Modelling.ipynb

# %%
# read in pickled clean data
cleaned_df = pd.read_pickle(f"data/ORD_USPTO/cleaned_data.pkl")

# %%
# read in the reaction classes
rxn_classes_filename = 'data/ORD_USPTO/classified_rxn.smi'

with open(rxn_classes_filename) as f:
    lines = f.readlines()
lines = [line.rstrip('\n') for line in lines] # remove the \n at the end of each line

# create df of the reaction classes
# 2 columns: mapped_rxn, rxn_classes
rxns = []
rxn_classes = []
for line in lines:
    try:
        rxn, rxn_class = line.split(' ')
        rxns += [rxn]
        rxn_classes += [rxn_class]
    except AttributeError:
        continue
    
rxn_classes_df = pd.DataFrame(list(zip(rxns, rxn_classes)),
               columns =['mapped_rxn', 'rxn_class'])
    

# %%
# combine the two dfs
data_df_temp = cleaned_df.merge(rxn_classes_df, how='inner', left_on='mapped_rxn_0', right_on='mapped_rxn')
len(data_df_temp)

# %%
# I used the following command to generate the rxn classification:
# ./namerxn -nomap data/mapped_rxn.smi data/classified_rxn.smi

# The -nomap I thought would mean that it wouldn't change the atom mapping, yet it clearly did...
# I'll just have to trust that namerxn didn't change the order of my reactions, and just append the reaction classes, and finally remove any reactions that couldn't be classified
data_df = cleaned_df.copy().reset_index(drop=True)
data_df['rxn_class'] = rxn_classes_df['rxn_class']
data_df = data_df.dropna(subset=['rxn_class'])
data_df.reset_index()
print(len(data_df))

# %%
# remove all the unclassified reactions, ie where rxn_class = '0.0'
remove_unclassified_rxn_data_df = data_df[~data_df.rxn_class.str.contains("0.0")]
print(len(remove_unclassified_rxn_data_df))

# %% [markdown]
# # Manual cleaning

# %% [markdown]
# ## Apply the cleaning that Alexander did

# %%
# print out all catalysts
#sorted(list(set(df['catalyst_0'].dropna())))

# initialize a dict that maps catalysts to the humanly cleaned smiles
catalyst_replacements = {}

catalyst_wrong = []
# All the data should have already been cleaned using rdkit.canonsmiles so I'm very surprised that there are some catalysts that are wrong. If you see any wrong catalysts, just remove them

# %%
# Add a catalyst to the catalyst_replacements dict
catalyst_replacements['CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].[Rh+3].[Rh+3]'] = 'CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].[Rh+2].[Rh+2]'
catalyst_replacements['[CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].[Rh+3]]'] = 'CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].[Rh+2].[Rh+2]'
catalyst_replacements['[CC(C)(C)[P]([Pd][P](C(C)(C)C)(C(C)(C)C)C(C)(C)C)(C(C)(C)C)C(C)(C)C]'] = 'CC(C)(C)[PH]([Pd][PH](C(C)(C)C)(C(C)(C)C)C(C)(C)C)(C(C)(C)C)C(C)(C)C'
catalyst_replacements['CCCC[N+](CCCC)(CCCC)CCCC.CCCC[N+](CCCC)(CCCC)CCCC.CCCC[N+](CCCC)(CCCC)CCCC.[Br-].[Br-].[Br-]'] = 'CCCC[N+](CCCC)(CCCC)CCCC.[Br-]'
catalyst_replacements['[CCO.CCO.CCO.CCO.[Ti]]'] = 'CCO[Ti](OCC)(OCC)OCC'
catalyst_replacements['[CC[O-].CC[O-].CC[O-].CC[O-].[Ti+4]]'] = 'CCO[Ti](OCC)(OCC)OCC'
catalyst_replacements['[Cl[Ni]Cl.c1ccc(P(CCCP(c2ccccc2)c2ccccc2)c2ccccc2)cc1]'] = 'Cl[Ni]1(Cl)[P](c2ccccc2)(c2ccccc2)CCC[P]1(c1ccccc1)c1ccccc1'
catalyst_replacements['[Cl[Pd](Cl)([P](c1ccccc1)(c1ccccc1)c1ccccc1)[P](c1ccccc1)(c1ccccc1)c1ccccc1]'] = 'Cl[Pd](Cl)([PH](c1ccccc1)(c1ccccc1)c1ccccc1)[PH](c1ccccc1)(c1ccccc1)c1ccccc1'
catalyst_replacements['[Cl[Pd+2](Cl)(Cl)Cl.[Na+].[Na+]]'] = 'Cl[Pd]Cl'
catalyst_replacements['Karstedt catalyst'] = 'C[Si](C)(C=C)O[Si](C)(C)C=C.[Pt]'
catalyst_replacements["Karstedt's catalyst"] = 'C[Si](C)(C=C)O[Si](C)(C)C=C.[Pt]'
catalyst_replacements['[O=C([O-])[O-].[Ag+2]]'] = 'O=C([O-])[O-].[Ag+].[Ag+]'
catalyst_replacements['[O=S(=O)([O-])[O-].[Ag+2]]'] = 'O=S(=O)([O-])[O-].[Ag+].[Ag+]'
catalyst_replacements['[O=[Ag-]]'] = 'O=[Ag]'
catalyst_replacements['[O=[Cu-]]'] = 'O=[Cu]'
catalyst_replacements['[Pd on-carbon]'] = '[C].[Pd]'
catalyst_replacements['[TEA]'] = 'OCCN(CCO)CCO'
catalyst_replacements['[Ti-superoxide]'] = 'O=[O-].[Ti]'
catalyst_replacements['[[Pd].c1ccc(P(c2ccccc2)c2ccccc2)cc1]'] = '[Pd].c1ccc(P(c2ccccc2)c2ccccc2)cc1.c1ccc(P(c2ccccc2)c2ccccc2)cc1.c1ccc(P(c2ccccc2)c2ccccc2)cc1.c1ccc(P(c2ccccc2)c2ccccc2)cc1'
catalyst_replacements['[c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd-4]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1]'] = 'c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1'
catalyst_replacements['[c1ccc([P]([Pd][P](c2ccccc2)(c2ccccc2)c2ccccc2)(c2ccccc2)c2ccccc2)cc1]'] = 'c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1'
catalyst_replacements['[c1ccc([P](c2ccccc2)(c2ccccc2)[Pd]([P](c2ccccc2)(c2ccccc2)c2ccccc2)([P](c2ccccc2)(c2ccccc2)c2ccccc2)[P](c2ccccc2)(c2ccccc2)c2ccccc2)cc1]'] = 'c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1'
catalyst_replacements['[sulfated tin oxide]'] = 'O=S(O[Sn])(O[Sn])O[Sn]'
catalyst_replacements['[tereakis(triphenylphosphine)palladium(0)]'] = 'c1ccc([PH](c2ccccc2)(c2ccccc2)[Pd]([PH](c2ccccc2)(c2ccccc2)c2ccccc2)([PH](c2ccccc2)(c2ccccc2)c2ccccc2)[PH](c2ccccc2)(c2ccccc2)c2ccccc2)cc1'
catalyst_replacements['[zeolite]'] = 'O=[Al]O[Al]=O.O=[Si]=O'

# %%
# add any wrong catalysts you spot, e.g.
catalyst_wrong += ['Catalyst A',
'catalyst',
'catalyst 1',
'catalyst A',
'catalyst VI',
'reaction mixture',
'same catalyst',
'solution']


# %%
# drop all rows that contain a 'catalyst_wrong
df2 = data_df[~data_df["catalyst_0"].isin(catalyst_wrong)]

# %%
# do the catalyst replacements that Alexander found
df3 = df2.replace(catalyst_replacements)

# %%
df3.reset_index(inplace=True)

# %%
count = 0
for i in range(len(data_df['reagents_0'])):
    r = data_df['reagents_0'][i]
    if r ==r:
        if 'pd' in r or 'Pd' in r or 'palladium' in r or 'Palladium' in r:
            count +=1
print('Number of Pd in the reagents columns: ', count )

# %%
# Quite a few of the rows have Pd as a reagent. Probably worth going through all of them, and if the value in reagent_0 is already in catalyst_0, then replace the reagent value with np.NaN
df3["reagents_0"] = df3.apply(lambda x: np.nan if (pd.notna(x["reagents_0"]) and pd.notna(x["catalyst_0"]) and x["reagents_0"] in x["catalyst_0"]) else x["reagents_0"], axis=1)
df3["reagents_1"] = df3.apply(lambda x: np.nan if (pd.notna(x["reagents_1"]) and pd.notna(x["catalyst_0"]) and x["reagents_1"] in x["catalyst_0"]) else x["reagents_1"], axis=1)


# %%
# That took care of a majority of the cases! Now there are only 9+7 cases left, just drop these rows
df3 = df3[df3["reagents_0"] != '[Pd]']
df3 = df3[df3["reagents_0"] != '[Pd+2]']
df3 = df3[df3["reagents_1"] != '[Pd]']
df3 = df3[df3["reagents_1"] != '[Pd+2]']
df3 = df3.reset_index(drop=True)



# %%
count = 0
for i in range(len(df3['reagents_1'])):
    r = df3['reagents_1'][i]
    if r ==r:
        if 'Pd' in r:
            print(r)
            count +=1
print('Number of Pd in the reagents columns: ', count )

# %%
len(df3)

# %% [markdown]
# ## Add a cluster column

# %%
df3['rxn_super_class'] = df3['rxn_class'].str.rsplit('.', expand=True)[0].astype(int)
test_df = df3['rxn_class'].str.rsplit(';', expand=True)
# 2.5% of reactions have been assigned 2 reaction classes. 3 or 4 reaction classes is very rare.

# %%
test_df

# %% [markdown]
# # Prepare fingerprints

# %%
from modelling_3 import calc_fp
from modelling_3 import calc_fp_individual
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from rdkit.rdBase import BlockLogs
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

# %%
%%time
num_cores = multiprocessing.cpu_count()
inputs = tqdm(df3['product_0'])
p0 = Parallel(n_jobs=num_cores)(delayed(calc_fp_individual)(i, 3, 512) for i in inputs)

inputs = tqdm(df3['product_1'])
p1 = Parallel(n_jobs=num_cores)(delayed(calc_fp_individual)(i, 3, 512) for i in inputs)

inputs = tqdm(df3['product_2'])
p2 = Parallel(n_jobs=num_cores)(delayed(calc_fp_individual)(i, 3, 512) for i in inputs)

inputs = tqdm(df3['product_3'])
p3 = Parallel(n_jobs=num_cores)(delayed(calc_fp_individual)(i, 3, 512) for i in inputs)

ar_p0 = np.array(p0)
ar_p1 = np.array(p1)
ar_p2 = np.array(p2)
ar_p3 = np.array(p3)

product_fp = ar_p0 + ar_p1 + ar_p2 + ar_p3

del p0, p1, p2, p3
del ar_p0, ar_p1, ar_p2, ar_p3

# %%
%%time
num_cores = multiprocessing.cpu_count()
inputs = tqdm(df3['reactant_0'])
r0 = Parallel(n_jobs=num_cores)(delayed(calc_fp_individual)(i, 3, 512) for i in inputs)

inputs = tqdm(df3['reactant_1'])
r1 = Parallel(n_jobs=num_cores)(delayed(calc_fp_individual)(i, 3, 512) for i in inputs)

inputs = tqdm(df3['reactant_2'])
r2 = Parallel(n_jobs=num_cores)(delayed(calc_fp_individual)(i, 3, 512) for i in inputs)

inputs = tqdm(df3['reactant_3'])
r3 = Parallel(n_jobs=num_cores)(delayed(calc_fp_individual)(i, 3, 512) for i in inputs)

ar_r0 = np.array(r0)
ar_r1 = np.array(r1)
ar_r2 = np.array(r2)
ar_r3 = np.array(r3)

react_fp = ar_r0 - ar_r1 - ar_r2 - ar_r3
del r0, r1, r2, r3
del ar_r0, ar_r1, ar_r2, ar_r3

# %%
rxn_diff_fp = product_fp - react_fp
rxn_diff_fp.shape

# %%
#save to pickle
# np.save("data/ORD_USPTO/USPTO_rxn_diff_fp.pkl", rxn_diff_fp)
# np.save("data/ORD_USPTO/USPTO_product_fp.pkl", product_fp)

# %% [markdown]
# # NN modelling

# %%
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics import Accuracy

# %% [markdown]
# ## Gao model

# %%
# https://pubs.acs.org/doi/full/10.1021/acscentsci.8b00357

# %%
#unpickle
# rxn_diff_fp = np.load("data/ORD_USPTO/USPTO_rxn_diff_fp.pkl.npy", allow_pickle=True)
# product_fp = np.load("data/ORD_USPTO/USPTO_product_fp.pkl.npy", allow_pickle=True)
# Run all cells in the "Read in data" section to get data_df

# find the data in df3

# %%
rxn_diff_fp.shape == product_fp.shape

# %%
df3['temperature_0'].count()
# almost no reactions have a temperature value

# %%
df3.columns

# %%
# prepare input for model

# np_input = np.concatenate((product_fp, rxn_diff_fp), axis=1)

# X = torch.from_numpy(np_input).float()
# # prepare output for model
# enc = OneHotEncoder()
# c_encoded = enc.fit_transform(df3[['catalyst_0']])
# s0_encoded = enc.fit_transform(df3[['solvent_0']])
# s1_encoded = enc.fit_transform(df3[['solvent_1']])
# r0_encoded = enc.fit_transform(df3[['reagents_0']])
# r1_encoded = enc.fit_transform(df3[['reagents_1']])


# c_encoded = c_encoded.toarray()
# s0_encoded = s0_encoded.toarray()
# s1_encoded = s1_encoded.toarray()
# r0_encoded = r0_encoded.toarray()
# r1_encoded = r1_encoded.toarray()


# y_c = torch.from_numpy(c_encoded).float()
# y_s0 = torch.from_numpy(s0_encoded).float()
# y_s1 = torch.from_numpy(s1_encoded).float()
# y_r0 = torch.from_numpy(r0_encoded).float()
# y_r1 = torch.from_numpy(r1_encoded).float()


# %%
rng = np.random.default_rng(12345)


indexes = np.arange(df3.shape[0])
rng.shuffle(indexes)

train_test_split = 0.5
train_val_split = 0.9

test_idx = indexes[int(df3.shape[0] * train_test_split):]
train_val_idx = indexes[:int(df3.shape[0] * train_test_split)]
train_idx = train_val_idx[:int(train_val_idx.shape[0] * train_val_split)]
val_idx = train_val_idx[int(train_val_idx.shape[0] * train_val_split):]

# %%
import typing

import sklearn
import pandas as pd
import torch


class GetDummies(sklearn.base.TransformerMixin):
    """Fast one-hot-encoder that makes use of pandas.get_dummies() safely
    on train/test splits.
    Taken from: https://dantegates.github.io/2018/05/04/a-fast-one-hot-encoder-with-sklearn-and-pandas.html
    """
    def __init__(self, dtypes=None):
        self.input_columns = None
        self.final_columns = None
        if dtypes is None:
            dtypes = [object, 'category']
        self.dtypes = dtypes

    def fit(self, X, y=None, dummy_na=True, **kwargs):
        self.input_columns = list(X.select_dtypes(self.dtypes).columns)
        X = pd.get_dummies(X, columns=self.input_columns, dummy_na=dummy_na)
        self.final_columns = X.columns
        return self
        
    def transform(self, X, y=None, **kwargs):
        X = pd.get_dummies(X, columns=self.input_columns)
        X_columns = X.columns
        # if columns in X had values not in the data set used during
        # fit add them and set to 0
        missing = set(self.final_columns) - set(X_columns)
        for c in missing:
            X[c] = 0
        # remove any new columns that may have resulted from values in
        # X that were not in the data set when fit
        return X[self.final_columns]
    
    def get_feature_names(self):
        return tuple(self.final_columns)


def apply_train_ohe_fit(df, train_idx, val_idx, tensor_func: typing.Optional[typing.Callable] = torch.Tensor):
    enc = GetDummies()
    _ = enc.fit(df.iloc[train_idx])
    _ohe = enc.transform(df)
    _tr, _val = _ohe.iloc[train_idx].values, _ohe.iloc[val_idx].values
    if tensor_func is not None:
        _tr, _val = tensor_func(_tr), tensor_func(_val)
    return _tr, _val, enc

np_input = np.concatenate((product_fp, rxn_diff_fp), axis=1)

train_input = torch.Tensor(np_input[train_idx])
val_input = torch.Tensor(np_input[val_idx])


train_catalyst, val_catalyst, cat_enc = apply_train_ohe_fit(df3[['catalyst_0']].fillna("NULL"), train_idx, val_idx)
train_solvent_0, val_solvent_0, sol0_enc = apply_train_ohe_fit(df3[['solvent_0']].fillna("NULL"), train_idx, val_idx)
train_solvent_1, val_solvent_1, sol1_enc = apply_train_ohe_fit(df3[['solvent_1']].fillna("NULL"), train_idx, val_idx)
train_reagents_0, val_reagents_0, reag0_enc = apply_train_ohe_fit(df3[['reagents_0']].fillna("NULL"), train_idx, val_idx)
train_reagents_1, val_reagents_1, reag1_enc = apply_train_ohe_fit(df3[['reagents_1']].fillna("NULL"), train_idx, val_idx)
train_temperature, val_temperature, temp_enc = apply_train_ohe_fit(df3[['temperature_0']].fillna(-1), train_idx, val_idx)

# %%
# # Split the data into train, validation, and test sets

# X_train, X_val_and_test, y_c_train, y_c_val_and_test, y_s0_train, y_s0_val_and_test, y_s1_train, y_s1_val_and_test, y_r0_train, y_r0_val_and_test, y_r1_train, y_r1_val_and_test = train_test_split(X, y_c, y_s0, y_s1, y_r0, y_r1, test_size=0.6, random_state=42)

# X_val, X_test, y_c_val, y_c_test, y_s0_val, y_s0_test, y_s1_val, y_s1_test, y_r0_val, y_r0_test, y_r1_val, y_r1_test = train_test_split(X_val_and_test, y_c_val_and_test, y_s0_val_and_test, y_s1_val_and_test, y_r0_val_and_test, y_r1_val_and_test, test_size=0.9, random_state=42)



# %%
X_train = train_input
y_c_train = train_catalyst
y_s0_train = train_solvent_0
y_s1_train = train_solvent_1
y_r0_train = train_reagents_0
y_r1_train = train_reagents_1
X_val = val_input
y_c_val = val_catalyst
y_s0_val = val_solvent_0
y_s1_val = val_solvent_1
y_r0_val = val_reagents_0
y_r1_val = val_reagents_1

# %%
val_reagents_1.shape

# %%
# use torch.nn
class NeuralNet(nn.Module):
    def __init__(self, df3, X_train):
        super(NeuralNet, self).__init__()
        
        self.c_num = train_catalyst.shape[1]
        self.s0_num = train_solvent_0.shape[1]
        self.s1_num = train_solvent_1.shape[1]
        self.r0_num = train_reagents_0.shape[1]
        self.r1_num = train_reagents_1.shape[1]
        
        # Architecture
        self.c_input = nn.Linear(X_train.shape[1], 1000)
        
        self.c_1000_1000 = nn.Linear(1000, 1000)
        self.c_1000_300 = nn.Linear(1000, 300)
        self.c_300_300 = nn.Linear(300, 300)
        
        # Catalyst_0 output layer
        self.c_output = nn.Linear(300, self.c_num)
        
        # print(f"{self.c_num=}")
        
        # s0
        self.s0_input = nn.Linear(self.c_num, 100)
        self.s0_1000_500 = nn.Linear(1000, 500)
        self.s0_600_300 = nn.Linear(600, 300)
        self.s0_300_300 = nn.Linear(300, 300)
        self.s0_output = nn.Linear(300, self.s0_num)
        
        # s1
        self.s1_input = nn.Linear(self.s0_num, 100)
        self.s1_700_300 = nn.Linear(700, 300)
        self.s1_300_300 = nn.Linear(300, 300)
        self.s1_output = nn.Linear(300, self.s1_num)
        
        # r0
        self.r0_input = nn.Linear(self.s1_num, 100)
        self.r0_800_300 = nn.Linear(800, 300)
        self.r0_300_300 = nn.Linear(300, 300)
        self.r0_output = nn.Linear(300, self.r0_num)

        # r1
        self.r1_input = nn.Linear(self.r0_num, 100)
        self.r1_900_300 = nn.Linear(900, 300)
        self.r1_300_300 = nn.Linear(300, 300)
        self.r1_output = nn.Linear(300, self.r1_num)


        # Activation functions etc. (stuff with parameters)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim = 1)
        
    def argmax_matrix(self, m):
        max_idx = torch.argmax(m, 1, keepdim=True)
        one_hot = torch.FloatTensor(m.shape)
        one_hot.zero_()
        one_hot.scatter_(1, max_idx, 1)
        return one_hot

    def forward(self, x, y_c, y_s0, y_s1, y_r0, y_r1, train):
        # print("A")
        x = self.c_input(x)
        x = self.relu(x)
        
        x = self.c_1000_1000(x)
        x = self.relu(x)
        # print("B")
        c_latent_1 = self.dropout(x)
        x = self.c_1000_300(c_latent_1)
        x = self.relu(x)
        x = self.c_300_300(x)
        x = self.tanh(x)
        # print("C")
        # print(f"{x.shape=}")
        
        # Catalyst_0 output
        c_out = self.c_output(x)
        #catalyst_out = self.softmax_cat(catalyst_out)
        # print("c1")
        # print(f"{c_out.shape=}")
        
        # Solvent_0 output
        if train:
            # print(f"{y_c.shape=}")
            s0_input = self.s0_input(y_c)
            # print(f"{s0_input.shape=}")
        else:
            # print("c2")
            # Feed in the probabilities:
            #solvent_input = self.solvent_input(catalyst_out)
            
            # convert softmax output to one-hot encoding
            c_one_hot = self.argmax_matrix(c_out)
            # print("c3")
            s0_input = self.s0_input(c_one_hot)
        # print("D")
        c_latent_2 = self.s0_1000_500(c_latent_1)
        c_s0_latent = torch.cat((c_latent_2, s0_input), 1)
        x = self.relu(c_s0_latent)
        x = self.s0_600_300(x)
        x = self.relu(x)
        x = self.s0_300_300(x)
        x = self.tanh(x)
        s0_out = self.s0_output(x)
        # print("E")

        # Solvent_1 output
        if train:
            s1_input = self.s1_input(y_s0)
        else:
            # Feed in the probabilities:
            #solvent_input = self.solvent_input(catalyst_out)
            
            # convert softmax output to one-hot encoding
            s0_one_hot = self.argmax_matrix(s0_out)
            s1_input = self.s1_input(s0_one_hot)

        c_s0_s1_latent = torch.cat((c_s0_latent, s1_input), 1)
        x = self.relu(c_s0_s1_latent)
        x = self.s1_700_300(x)
        x = self.relu(x)
        x = self.s1_300_300(x)
        x = self.tanh(x)
        s1_out = self.s1_output(x)
        
        # Reagent_0 output
        if train:
            r0_input = self.r0_input(y_s1)
        else:
            # Feed in the probabilities:
            #solvent_input = self.solvent_input(catalyst_out)
            
            # convert softmax output to one-hot encoding
            s1_one_hot = self.argmax_matrix(s1_out)
            r0_input = self.r0_input(s1_one_hot)

        c_s0_s1_r0_latent = torch.cat((c_s0_s1_latent, r0_input), 1)
        x = self.relu(c_s0_s1_r0_latent)
        x = self.r0_800_300(x)
        x = self.relu(x)
        x = self.r0_300_300(x)
        x = self.tanh(x)
        r0_out = self.r0_output(x)

        # Reagent_1 output
        if train:
            r1_input = self.r1_input(y_r0)
        else:
            # Feed in the probabilities:
            #solvent_input = self.solvent_input(catalyst_out)
            
            # convert softmax output to one-hot encoding
            r0_one_hot = self.argmax_matrix(r0_out)
            r1_input = self.r1_input(r0_one_hot)

        c_s0_s1_r0_r1_latent = torch.cat((c_s0_s1_r0_latent, r1_input), 1)
        x = self.relu(c_s0_s1_r0_r1_latent)
        x = self.r1_900_300(x)
        x = self.relu(x)
        x = self.r1_300_300(x)
        x = self.tanh(x)
        r1_out = self.r1_output(x)

        
        return c_out, s0_out, s1_out, r0_out, r1_out


# %%
model = NeuralNet(df3=df3, X_train=train_input)
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Define the number of training iterations
num_epochs = 100

# %%
def top_k_accuracy(k, predicted, actual):
    accuracy = Accuracy(task="multiclass", num_classes=len(predicted[1]), top_k=k)
    acc = accuracy(predicted, torch.argmax(actual, dim=1))
    return acc

# %%
# Initialize lists to store the train and test loss
train_loss = []
test_loss = []

top_1_c_train = []
top_3_c_train = []
top_1_s0_train = []
top_3_s0_train = []
top_1_s1_train = []
top_3_s1_train = []
top_1_r0_train = []
top_3_r0_train = []
top_1_r1_train = []
top_3_r1_train = []

top_1_c_test = []
top_3_c_test = []
top_1_s0_test = []
top_3_s0_test = []
top_1_s1_test = []
top_3_s1_test = []
top_1_r0_test = []
top_3_r0_test = []
top_1_r1_test = []
top_3_r1_test = []



# Define the batch size
batch_size = 256
num_batches = int(X_train.shape[0] / batch_size)

# Create data loaders for the training data
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_c_train, y_s0_train, y_s1_train, y_r0_train, y_r1_train),
    batch_size=batch_size,
    shuffle=True
)


# Create data loader for test data
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_val, y_c_val, y_s0_val, y_s1_val, y_r0_val, y_r1_val),
    batch_size=batch_size,
    shuffle=True
)



# Training the model
for epoch in range(num_epochs):
    train_loss_batch = []
    
    top_1_c_train_batch = []
    top_3_c_train_batch = []
    top_1_s0_train_batch = []
    top_3_s0_train_batch = []
    top_1_s1_train_batch = []
    top_3_s1_train_batch = []
    top_1_r0_train_batch = []
    top_3_r0_train_batch = []
    top_1_r1_train_batch = []
    top_3_r1_train_batch = []
    
    for X_batch, y_c_batch, y_s0_batch, y_s1_batch, y_r0_batch, y_r1_batch in tqdm(train_loader):
        # Forward pass
        # print(X_batch.shape)
        # print(y_c_batch.shape)

        c_out, s0_out, s1_out, r0_out, r1_out = model(X_batch, y_c_batch, y_s0_batch, y_s1_batch, y_r0_batch, y_r1_batch, train = True)
        
        # Compute the train loss
        loss_c = criterion(c_out, y_c_batch)
        loss_s0 = criterion(s0_out, y_s0_batch)
        loss_s1 = criterion(s1_out, y_s1_batch)
        loss_r0 = criterion(r0_out, y_r0_batch)
        loss_r1 = criterion(r1_out, y_r1_batch)
        
        loss = loss_c + loss_s0 + loss_s1 + loss_r0 + loss_r1
        train_loss_batch.append(loss.item())
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Update the parameters
        optimizer.step()
        
        # Compute accuracy for c
        top_1_c_train_batch += [top_k_accuracy(1, c_out, y_c_batch)]
        top_3_c_train_batch += [top_k_accuracy(3, c_out, y_c_batch)]
        
        # Compute accuracy for s0
        top_1_s0_train_batch += [top_k_accuracy(1, s0_out, y_s0_batch)]
        top_3_s0_train_batch += [top_k_accuracy(3, s0_out, y_s0_batch)]
        
        # Compute accuracy for s1
        top_1_s1_train_batch += [top_k_accuracy(1, s1_out, y_s1_batch)]
        top_3_s1_train_batch += [top_k_accuracy(3, s1_out, y_s1_batch)]

        # Compute accuracy for r0
        top_1_r0_train_batch += [top_k_accuracy(1, r0_out, y_r0_batch)]
        top_3_r0_train_batch += [top_k_accuracy(3, r0_out, y_r0_batch)]
    
        # Compute accuracy for r1
        top_1_r1_train_batch += [top_k_accuracy(1, r1_out, y_r1_batch)]
        top_3_r1_train_batch += [top_k_accuracy(3, r1_out, y_r1_batch)]
                
    
    train_loss += [np.mean(train_loss_batch)]
        
    top_1_c_train += [np.mean(top_1_c_train_batch)]
    top_3_c_train += [np.mean(top_3_c_train_batch)]
    
    top_1_s0_train += [np.mean(top_1_s0_train_batch)]
    top_3_s0_train += [np.mean(top_3_s0_train_batch)]
    
    top_1_s1_train += [np.mean(top_1_s1_train_batch)]
    top_3_s1_train += [np.mean(top_3_s1_train_batch)]

    top_1_r0_train += [np.mean(top_1_r0_train_batch)]
    top_3_r0_train += [np.mean(top_3_r0_train_batch)]

    top_1_r1_train += [np.mean(top_1_r1_train_batch)]
    top_3_r1_train += [np.mean(top_3_r1_train_batch)]    
        
        
    # Compute the test loss
    with torch.no_grad():
        test_loss_batch = []
        
        top_1_c_test_batch = []
        top_3_c_test_batch = []
        top_1_s0_test_batch = []
        top_3_s0_test_batch = []
        top_1_s1_test_batch = []
        top_3_s1_test_batch = []
        top_1_r0_test_batch = []
        top_3_r0_test_batch = []
        top_1_r1_test_batch = []
        top_3_r1_test_batch = []
        
        for X_batch, y_c_batch, y_s0_batch, y_s1_batch, y_r0_batch, y_r1_batch in test_loader:
            c_out, s0_out, s1_out, r0_out, r1_out = model(X_batch, y_c_batch, y_s0_batch, y_s1_batch, y_r0_batch, y_r1_batch, train = True)
            
            # Compute the test loss
            loss_c = criterion(c_out, y_c_batch)
            loss_s0 = criterion(s0_out, y_s0_batch)
            loss_s1 = criterion(s1_out, y_s1_batch)
            loss_r0 = criterion(r0_out, y_r0_batch)
            loss_r1 = criterion(r1_out, y_r1_batch)
            
            loss = loss_c + loss_s0 + loss_s1 + loss_r0 + loss_r1
            test_loss_batch.append(loss.item())
            
            # Compute accuracy for c
            top_1_c_test_batch += [top_k_accuracy(1, c_out, y_c_batch)]
            top_3_c_test_batch += [top_k_accuracy(3, c_out, y_c_batch)]
            
            # Compute accuracy for s0
            top_1_s0_test_batch += [top_k_accuracy(1, s0_out, y_s0_batch)]
            top_3_s0_test_batch += [top_k_accuracy(3, s0_out, y_s0_batch)]
            
            # Compute accuracy for s1
            top_1_s1_test_batch += [top_k_accuracy(1, s1_out, y_s1_batch)]
            top_3_s1_test_batch += [top_k_accuracy(3, s1_out, y_s1_batch)]

            # Compute accuracy for r0
            top_1_r0_test_batch += [top_k_accuracy(1, r0_out, y_r0_batch)]
            top_3_r0_test_batch += [top_k_accuracy(3, r0_out, y_r0_batch)]
        
            # Compute accuracy for r1
            top_1_r1_test_batch += [top_k_accuracy(1, r1_out, y_r1_batch)]
            top_3_r1_test_batch += [top_k_accuracy(3, r1_out, y_r1_batch)]
                
    
    test_loss += [np.mean(test_loss_batch)]
        
    top_1_c_test += [np.mean(top_1_c_test_batch)]
    top_3_c_test += [np.mean(top_3_c_test_batch)]
    
    top_1_s0_test += [np.mean(top_1_s0_test_batch)]
    top_3_s0_test += [np.mean(top_3_s0_test_batch)]
    
    top_1_s1_test += [np.mean(top_1_s1_test_batch)]
    top_3_s1_test += [np.mean(top_3_s1_test_batch)]

    top_1_r0_test += [np.mean(top_1_r0_test_batch)]
    top_3_r0_test += [np.mean(top_3_r0_test_batch)]

    top_1_r1_test += [np.mean(top_1_r1_test_batch)]
    top_3_r1_test += [np.mean(top_3_r1_test_batch)]  


            
    # Print train loss and accuracy
    print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss[-1]))
    print('Epoch [{}/{}], Test Loss: {:.4f}'.format(epoch+1, num_epochs, test_loss[-1]))




# %%
# Plot the train and test loss
plt.plot(train_loss, label='train loss')
plt.plot(test_loss, label='test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()

# Plot the train and test accuracy
# Calyst
plt.plot(top_1_c_test, label='top_1_c_test')
plt.plot(top_3_c_test, label='top_3_c_test')
plt.plot(top_1_c_train, label='top_1_c_train')
plt.plot(top_3_c_train, label='top_3_c_train')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.clf()

# Plot the train and test accuracy
# S1ent
plt.plot(top_1_s1_test, label='top_1_s1_test')
plt.plot(top_3_s1_test, label='top_3_s1_test')
plt.plot(top_1_s1_train, label='top_1_s1_train')
plt.plot(top_3_s1_train, label='top_3_s1_train')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.clf()





# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %% [markdown]
# ## Other models

# %%
import torch
import torchmetrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tqdm import trange

from modelling_3 import FullyConnectedReactionModel
from modelling_3 import train_loop

e=70
batch_size=0.07
lr=1e-4


# %%
#unpickle
rxn_diff_fp = np.load("data/ORD_USPTO/USPTO_rxn_diff_fp.pkl.npy", allow_pickle=True)
# Run all cells in the "Read in data" section to get data_df

# %%
rxn_diff_fp.shape

# %%
data_df.columns

# %%
len(data_df)

# %%
# prep the data
cat = np.array(data_df['catalyst_0'])
# Do the one-hot encoding
enc = OneHotEncoder(handle_unknown='ignore')
cat_reshaped = cat.reshape(-1, 1)
_ = enc.fit(cat_reshaped)

cat_ohe = enc.transform(cat.reshape(-1, 1)).toarray()

rxn_diff_fp_train, rxn_diff_fp_val, cat_ohe_train, cat_ohe_val = train_test_split(rxn_diff_fp, cat_ohe, test_size=0.2, random_state=42)

# %% [markdown]
# ## Big NN (full sharing)

# %%
# A data through one model for all reactions
# rxn_diff_fp width -> ohe width

x_train = torch.Tensor(rxn_diff_fp_train)
y_train = torch.Tensor(cat_ohe_train)
x_val = torch.Tensor(rxn_diff_fp_val)
y_val = torch.Tensor(cat_ohe_val)
fcrm = FullyConnectedReactionModel(
    input_dim=x_train.shape[1],
    hidden_dims=[600, 600],
    target_dim=y_train.shape[1],
    hidden_act=torch.nn.ReLU, 
    output_act=torch.nn.Identity, 
    use_batchnorm=True, 
    dropout_prob=0.2,
)

optimizer = torch.optim.Adam(fcrm.parameters(), lr=lr)
scheduler = None  # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)
early_stopper = None  # EarlyStopper(patience=50, min_delta=0, verbose=True)

hist = train_loop(fcrm, x_train, y_train, epochs=e, batch_size=batch_size, loss_fn=torch.nn.CrossEntropyLoss(), optimizer=optimizer, report_freq=1, x_val=x_val, y_val=y_val, scheduler=scheduler, early_stopper=early_stopper)

# %%
f,ax=plt.subplots(1,4, figsize=(15, 5))
hist.loc[:, (slice(None), "loss")].droplevel(level=1,axis=1).plot(title="Cross Entropy (Loss)", ax=ax[0])
hist.loc[:, (slice(None), "acc")].droplevel(level=1,axis=1).plot(title="Accuracy", ax=ax[1])
hist.loc[:, (slice(None), "acc_top3")].droplevel(level=1,axis=1).plot(title="Top 3 Accuracy", ax=ax[2])
hist.loc[:, (slice(None), "acc_top5")].droplevel(level=1,axis=1).plot(title="Top 5 Accuracy", ax=ax[3])
f.suptitle("All Data")

# %% [markdown]
# ## 1 model per cluster

# %%
# Data through one model per reaction cluster
# rxn_diff_fp width -> ohe width

train_clusters = Kmean.predict(rxn_diff_fp_train)
val_clusters = Kmean.predict(rxn_diff_fp_val)

cluster_models = {}
cluster_histories = {}

for cluster in np.unique(train_clusters):
    print(cluster)
    x_train = torch.Tensor(rxn_diff_fp_train[train_clusters == cluster])
    y_train = torch.Tensor(reag1_ohe_train[train_clusters == cluster])
    x_val = torch.Tensor(rxn_diff_fp_val[val_clusters == cluster])
    y_val = torch.Tensor(reag1_ohe_val[val_clusters == cluster])
    fcrm = FullyConnectedReactionModel(
        input_dim=x_train.shape[1],
        hidden_dims=[600,600],
        target_dim=y_train.shape[1],
        hidden_act=torch.nn.ReLU, 
        output_act=torch.nn.Identity, 
        use_batchnorm=True, 
        dropout_prob=0.2,
    )

    optimizer=torch.optim.Adam(fcrm.parameters(), lr=lr)
    scheduler = None#torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)
    early_stopper = None#EarlyStopper(patience=50, min_delta=0, verbose=True)

    hist = train_loop(fcrm, x_train, y_train, epochs=e, batch_size=batch_size, loss_fn=torch.nn.CrossEntropyLoss(), optimizer=optimizer, report_freq=1, x_val=x_val, y_val=y_val, scheduler=scheduler, early_stopper=early_stopper)
    cluster_models[cluster] = fcrm
    cluster_histories[cluster] = hist

# %%
y_train = torch.Tensor(reag1_ohe_train)
y_val = torch.Tensor(reag1_ohe_val)
pred_train = torch.zeros_like(y_train)
pred_val = torch.zeros_like(y_val)
for cluster in np.unique(train_clusters):
    x_train = torch.Tensor(rxn_diff_fp_train[train_clusters == cluster])
    x_val = torch.Tensor(rxn_diff_fp_val[val_clusters == cluster])
    cluster_pred_train = cluster_models[cluster](x_train, training=False)
    cluster_pred_val = cluster_models[cluster](x_val, training=False)
    pred_train[train_clusters == cluster] = cluster_pred_train
    pred_val[val_clusters == cluster] = cluster_pred_val

acc_metric_top1 = torchmetrics.Accuracy(task="multiclass", num_classes=y_train.shape[1], top_k=1)
acc_metric_top3 = torchmetrics.Accuracy(task="multiclass", num_classes=y_train.shape[1], top_k=3)
acc_metric_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=y_train.shape[1], top_k=5)

train_acc, train_acc_top3, train_acc_top5 = acc_metric_top1(pred_train, y_train.argmax(axis=1)).item(), acc_metric_top3(pred_train, y_train.argmax(axis=1)).item(), acc_metric_top5(pred_train, y_train.argmax(axis=1)).item()
val_acc, val_acc_top3, val_acc_top5 = acc_metric_top1(pred_val, y_val.argmax(axis=1)).item(), acc_metric_top3(pred_val, y_val.argmax(axis=1)).item(), acc_metric_top5(pred_val, y_val.argmax(axis=1)).item()

print(f"Train      [ Acc (Top 1): {train_acc:.5f} | Acc (Top 3): {train_acc_top3:.5f} | Acc (Top 5): {train_acc_top5: .5f}]")
print(f"Validation [ Acc (Top 1): {val_acc:.5f} | Acc (Top 3): {val_acc_top3:.5f} | Acc (Top 5): {val_acc_top5: .5f}]")

