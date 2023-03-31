import typing

from rdkit import Chem as rdkit_Chem
from rdkit.rdBase import BlockLogs as rdkit_BlockLogs

from orderly.types import *


def remove_mapping_info_and_canonicalise_smiles(
    molecule_identifier: MOLECULE_IDENTIFIER,
) -> typing.Optional[SMILES]:
    # This function can handle smiles both with and without mapping info
    _ = rdkit_BlockLogs()
    # remove mapping info and canonicalsie the molecule_identifier at the same time
    # converting to mol and back canonicalises the molecule_identifier string
    try:
        m = rdkit_Chem.MolFromSmiles(molecule_identifier)
        for atom in m.GetAtoms():
            atom.SetAtomMapNum(0)
        return rdkit_Chem.MolToSmiles(m)
    except AttributeError:
        return None


def canonicalise_smiles(molecule_identifier: MOLECULE_IDENTIFIER) -> typing.Optional[SMILES]:
    _ = rdkit_BlockLogs()
    # remove mapping info and canonicalsie the molecule_identifier at the same time
    # converting to mol and back canonicalises the molecule_identifier string
    try:
        return rdkit_Chem.CanonSmiles(molecule_identifier)
    except AttributeError:
        return None
    except Exception as e:
        # raise e
        return None


def get_canonicalised_smiles(molecule_identifier: MOLECULE_IDENTIFIER, is_mapped: bool = False) -> typing.Optional[SMILES]:
    # attempts to remove mapping info and canonicalise a smiles string and if it fails, returns the name whilst adding to a list of non smiles names
    # molecule_identifier: string, that is a smiles or an english name of the molecule
    if is_mapped:
        return remove_mapping_info_and_canonicalise_smiles(
            molecule_identifier
        )
    return canonicalise_smiles(molecule_identifier)