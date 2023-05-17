from tqdm import tqdm
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.rdBase import BlockLogs


def calc_fp(lst: list, radius=3, nBits=2048):
    # Usage:
    # radius = 3
    # nBits = 2048
    # p0 = calc_fp(data_df['product_0'][:10000], radius=radius, nBits=nBits)

    ans = []
    for i in tqdm(lst):
        # convert to mole object
        try:
            block = BlockLogs()
            mol = Chem.MolFromSmiles(i)
            # We are using hashed fingerprint, becasue an unhased FP has length: 4294967295
            fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=nBits)
            array = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, array)
            ans += [array]
        except:
            ans += [np.zeros((nBits,), dtype=int)]
    return ans


def calc_fp_individual(smiles: str, radius=3, nBits=2048):
    # ans = []
    try:
        block = BlockLogs()
        mol = Chem.MolFromSmiles(smiles)
        # We are using hashed fingerprint, becasue an unhased FP has length: 4294967295
        fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=nBits)
        array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, array)
        return array
    except:
        array = np.zeros((nBits,), dtype=int)
        return array
