import logging
import os
import typing
import datetime
import pathlib
import pickle
import click

import pandas as pd
import numpy as np
import tqdm
import tqdm.contrib.logging

from rdkit import Chem as rdkit_Chem
from rdkit.rdBase import BlockLogs as rdkit_BlockLogs

import orderly.extract.extractor
import orderly.extract.canonicalise
import orderly.extract.defaults
import orderly.data

from orderly.types import *

LOG = logging.getLogger(__name__)


def get_file_names(
    directory: pathlib.Path,
    file_ending: str = ".pb.gz",
) -> typing.List[pathlib.Path]:
    """
    Goes into the ord data directory and for each folder extracts the file path of all sub data files with the file ending
    """

    files = []
    for i in directory.glob("./*"):
        for j in i.glob(f"./*{file_ending}"):
            files.append(j)

    return sorted(
        files
    )  # sort just so that there is no randomness in order of processing


def merge_pickled_mol_names(
    molecule_names_path: pathlib.Path = pathlib.Path("data/orderly/molecule_names"),
    output_file_path: pathlib.Path = pathlib.Path(
        "data/orderly/all_molecule_names.pkl"
    ),
    overwrite: bool = True,
    molecule_names_file_ending: str = ".pkl",
):
    """
    Merges all the pickle files containing molecule non-smiles identifiers (typically english names) into one file.
    """
    if output_file_path.suffix != ".pkl":
        raise ValueError(
            f"The file extension for {output_file_path=} is expected to be .pkl not {output_file_path.suffix}"
        )
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        if output_file_path.exists():
            e = FileExistsError(
                f"{output_file_path} exists, with {overwrite=}, we expect the file to not exist."
            )
            LOG.error(e)
            raise e

    full_lst = []
    for f in molecule_names_path.glob(f"./*{molecule_names_file_ending}"):
        full_lst += pd.read_pickle(f)

    unique_molecule_names = list(set(full_lst))

    # pickle the list
    with open(output_file_path, "wb") as f:
        pickle.dump(unique_molecule_names, f)
    LOG.info(f"Pickled list of unique molecule names at {output_file_path=}")


def build_solvents_set_and_dict(
    solvents_path: typing.Optional[pathlib.Path] = None,
) -> typing.Tuple[typing.Set, typing.Dict]:
    """
    Builds a set of canonical smiles strings for all solvents (used to identify which agents are solvents) and a dictionary of solvent names to canonical smiles strings (for name resolution)
    """
    solvents = orderly.data.get_solvents(path=solvents_path)

    solvents["canonical_smiles"] = solvents["smiles"].apply(
        orderly.extract.canonicalise.get_canonicalised_smiles
    )

    solvents_set = set(solvents["canonical_smiles"])

    # Combine the lists into a sequence of key-value pairs
    key_value_pairs = zip(
        list(solvents["stenutz_name"]) + list(solvents["cosmo_name"]),
        list(solvents["canonical_smiles"]) + list(solvents["canonical_smiles"]),
    )

    # Create a dictionary from the sequence
    solvents_dict = dict(key_value_pairs)

    return solvents_set, solvents_dict


def build_replacements(
    molecule_replacements: typing.Optional[typing.Dict[str, str]] = None,
    molecule_str_force_nones: typing.Optional[typing.List[str]] = None,
) -> typing.Dict[str, typing.Optional[str]]:
    """
    Builds dictionary mapping english name molecule identifiers to canonical smiles. Dict is based on manually curated list.
    """
    _ = rdkit_BlockLogs()  # removes excessive warnings

    if molecule_replacements is None:
        molecule_replacements = orderly.extract.defaults.get_molecule_replacements()

    # Iterate over the dictionary and canonicalize each SMILES string
    for key, value in molecule_replacements.items():
        mol = rdkit_Chem.MolFromSmiles(value)
        if mol is not None:
            molecule_replacements[key] = rdkit_Chem.MolToSmiles(mol)

    if molecule_str_force_nones is None:
        molecule_str_force_nones = (
            orderly.extract.defaults.get_molecule_str_force_nones()
        )

    for molecule_str in molecule_str_force_nones:
        molecule_replacements[molecule_str] = None

    LOG.debug("Got molecule replacements")
    return molecule_replacements


def get_manual_replacements_dict(
    molecule_replacements: typing.Optional[
        typing.Dict[MOLECULE_IDENTIFIER, CANON_SMILES]
    ] = None,
    molecule_str_force_nones: typing.Optional[typing.List[MOLECULE_IDENTIFIER]] = None,
    solvents_path: typing.Optional[pathlib.Path] = None,
):
    """
    Combines manually curated dictioary of molecule names to canonical smiles strings with the dictionary of solvent names to canonical smiles strings.
    """
    manual_replacements_dict = build_replacements(
        molecule_replacements=molecule_replacements,
        molecule_str_force_nones=molecule_str_force_nones,
    )
    solvents_dict = orderly.data.get_solvents_dict(path=solvents_path)
    manual_replacements_dict.update(solvents_dict)
    return manual_replacements_dict


def extract(
    output_path: pathlib.Path,
    file,
    trust_labelling,
    manual_replacements_dict,
    solvents_set,
    pickled_data_folder: str = "pickled_data",
    molecule_names_folder: str = "molecule_names",
    name_contains_substring: typing.Optional[str] = None,
    inverse_substring: bool = False,
    overwrite: bool = True,
):
    """
    Extract information from an ORD file.
    """
    LOG.debug(f"Attempting extraction for {file}")
    instance = orderly.extract.extractor.OrdExtractor(
        ord_file_path=file,
        trust_labelling=trust_labelling,
        manual_replacements_dict=manual_replacements_dict,
        solvents_set=solvents_set,
        contains_substring=name_contains_substring,
        inverse_contains_substring=inverse_substring,
    )
    if instance.full_df is None:
        LOG.debug(f"Skipping extraction for {file}")
        return

    filename = instance.filename
    LOG.info(f"Completed extraction for {file}: {filename}")

    df_path = output_path / pickled_data_folder / f"{filename}.pkl"
    molecule_names_path = (
        output_path / molecule_names_folder / f"molecules_{filename}.pkl"
    )
    if not overwrite:
        if df_path.exists():
            e = FileExistsError(
                f"Trying to overwrite {df_path} which exists, overwrite must be true to do this"
            )
            LOG.error(e)
            raise e
        if molecule_names_path.exists():
            e = FileExistsError(
                f"Trying to overwrite {molecule_names_path} which exists, overwrite must be true to do this"
            )
            LOG.error(e)
            raise e

    instance.full_df.to_pickle(df_path)
    LOG.debug(f"Saved df at {df_path}")

    # list of the names used for molecules, as opposed to SMILES strings
    # save the non_smiles_names_list to pickle file
    with open(
        output_path / molecule_names_folder / f"molecules_{filename}.pkl", "wb"
    ) as f:
        pickle.dump(instance.non_smiles_names_list, f)
    LOG.debug(f"Saves molecule names for {filename} at {molecule_names_path}")


@click.command()
@click.option(
    "--data_path",
    type=str,
    default="data/ord/",
    show_default=True,
    help="The path of the folder that contains the ORD data",
)
@click.option(
    "--ord_file_ending",
    type=str,
    default=".pb.gz",
    help="The file ending for the ord data, if empty will extract all files in the folder",
    show_default=True,
)
@click.option(
    "--trust_labelling",
    type=bool,
    default=False,
    show_default=False,
    help="""
- If True, maintain the labelling and ordering of the original data.
- If False: Trust the mapped reaction more than the labelled data. A reaction string should be of the form reactants>agents>products; however, agents (particularly reagents) may sometimes appear as reactants on the LHS, so any molecules on the LHS we re-label as a reagent if it (i) does not contain any atom mapped atoms, (ii) the molecule appears on both the LHS and RHS (ie it is unreacted). Note that the original labelling is trusted if the reaction is not mapped. The agents list consists of catalysts, reagents and solvents; any molecules that occur in the set of solvents are extracted from the agents list and re-labelled as solvents, while the remaining molecules remain labelled as agents. Then the list of agents and solvents is sorted alphabetically, and finally any molecules that contain a metal were moved to the front of the agents list; ideally these lists be sorted by using chemical reasoning (e.g. by amount or importance), however this data doesn't exist, so we sort alphabetically and move metal containing molecules to the front (since its likely to be a catalyst) to at least add some order, albeit an arbitrary one. Prior work indicates that sequential prediction of conditions outperforms predicting all conditions in a single output layer (https://doi.org/10.1021/acscentsci.8b00357), so ordering may be helpful.""",
)
@click.option(
    "--output_path",
    type=str,
    default="data/orderly/",
    show_default=True,
    help="The path to the folder than will contain the pickled_data_folder and molecule_names_folder",
)
@click.option(
    "--pickled_data_folder",
    type=str,
    default="pickled_data",
    show_default=True,
    help="The name of folder than contains the pickle data structures",
)
@click.option(
    "--solvents_path",
    type=str,
    default="default",
    show_default=True,
    help="The path to the solvents csv, if None will use the default",
)
@click.option(
    "--molecule_names_folder",
    type=str,
    default="molecule_names",
    show_default=True,
    help="The name of the folder that contains the molecule_name pickles per folder",
)
@click.option(
    "--merged_molecules_file",
    type=str,
    default="all_molecule_names.pkl",
    show_default=True,
    help="The path to the merged_molecules pickle file",
)
@click.option(
    "--use_multiprocessing",
    type=bool,
    default=True,
    show_default=True,
    help="Boolean to make the processing of each ORD file done with multiprocessing",
)
@click.option(
    "--name_contains_substring",
    type=str,
    default="uspto",
    show_default=True,
    help="A filter for the ORD file names, will only extract files that includes the str. If left empty will not search for anything. For example 'uspto' grabs only uspto data",
)
@click.option(
    "--inverse_substring",
    type=bool,
    default=False,
    show_default=True,
    help="Inversed the name contains substring, so name_contains_substring='uspto' & inverse_substring=True will exclude names with uspto in",
)
@click.option(
    "--overwrite",
    type=bool,
    default=True,
    show_default=True,
    help="If true, will overwrite existing files, else will through an error if a file exists",
)
def main_click(
    data_path: str,
    ord_file_ending: str,
    trust_labelling: bool,
    output_path: str,
    pickled_data_folder: str,
    solvents_path: str,
    molecule_names_folder: str,
    merged_molecules_file: str,
    use_multiprocessing: bool,
    name_contains_substring: str,
    inverse_substring: bool,
    overwrite: bool,
):
    """
    After downloading the dataset from ORD, this script will extract the data and write it to pickle files.
        Example:


    Args:

    1) data_path: str
        - Path to the folder that contains the ORD data
    2) ord_file_ending: str
        - the file ending of the ord data typically ".pb.gz"
    3) trust_labelling: Bool
        - If True, maintain the labelling and ordering of the original data.
        - If False: Trust the mapped reaction more than the labelled data. A reaction string should be of the form reactants>agents>products; however, agents (particularly reagents) may sometimes appear as reactants on the LHS, so any molecules on the LHS we re-label as a reagent if it (i) does not contain any atom mapped atoms, (ii) the molecule appears on both the LHS and RHS (ie it is unreacted). Note that the original labelling is trusted if the reaction is not mapped. The agents list consists of catalysts, reagents and solvents; any molecules that occur in the set of solvents are extracted from the agents list and re-labelled as solvents, while the remaining molecules remain labelled as agents. Then the list of agents and solvents is sorted alphabetically, and finally any molecules that contain a metal were moved to the front of the agents list; ideally these lists be sorted by using chemical reasoning (e.g. by amount or importance), however this data doesn't exist, so we sort alphabetically and move metal containing molecules to the front (since its likely to be a catalyst) to at least add some order, albeit an arbitrary one. Prior work indicates that sequential prediction of conditions outperforms predicting all conditions in a single output layer (https://doi.org/10.1021/acscentsci.8b00357), so ordering may be helpful.
    4) output_path: str
        - The path to the folder than will contain the pickled_data_folder and molecule_names_folder
    5) pickled_data_folder: str
        - The name of folder than contains the pickle data structures
    6) solvents_path: typing.Optional[str]
        - The path to the solvents csv, if None will use the default
    7) molecule_names_folder: str
        - The name of the folder that contains the molecule_name pickles per folder
    8) merged_molecules_file: str
        - The path to the merged_molecules pickle file
    9) use_multiprocessing: bool
        - Boolean to make the processing of each ORD file done with multiprocessing
    10) name_contains_substring: typing.Optional[str]
        - A filter for the ORD file names, will only extract files that includes the str. If left empty will not search for anything. For example 'uspto' grabs only uspto data
    11) inverse_substring: bool
        - Inversed the name contains substring, so name_contains_substring='uspto' & inverse_substring=True will exclude names with uspto in
    12) overwrite: bool
        - If true, will overwrite existing files, else will through an error if a file exists.


    Functionality:

    1) USPTO data extracted from ORD comes in a large number of files (.pd.gz) batched in a large number of sub-folders. First step is to extract all filepaths that contain USPTO data (by checking whether the string 'uspto' is contained in the filename).
    2) Iterated over all filepaths to extract the following data:
        - The mapped reaction (unchanged)
        - Reactants and products (extracted from the mapped reaction)
        - Reagents: some reagents were extracted from the mapped reaction (if between the >> symbols) while other reagents were labelled as reagents in ORD
        - Solvents and catalysts: labelled as such in ORD
        - Temperature: All temperatures were converted to Celcius. If only the control type was specified, the following mapping was used: 'AMBIENT' -> 25, 'ICE_BATH' -> 0, 'DRY_ICE' -> -78.5, 'LIQUID_NITROGEN' -> -196.
        - Time: All times were converted to hours.
        - Yield (for each product): The PERCENTAGEYIELD was preferred, but if this was not available, the CALCULATEDPERCENTYIELD was used instead. If neither was available, the value was set to np.nan.
        - Procedure_details: Plain text string describing the procedure in English.
    3) Canonicalisation and light cleaning
        - All SMILES strings were canonicalised using RDKit.
        - A "replacements dictionary" was created to replace common names with their corresponding SMILES strings. This dictionary was created by iterating over the most common names in the dataset and replacing them with their corresponding SMILES strings. This was done semi-manually, with some dictionary entries coming from solvents.csv and others being added within the script (in the build_replacements function; mainly concerning catalysts).
        - The final light cleaning step depends on the value of trust_labelling (see above, in the Args section).
        - Reactions will only be added if the reactants and products are different (i.e. no crystalisation reactions etc.)
    4) Build a pandas DataFrame from this data (one for each ORD file), and save each as a pickle file
    5) Create a list of all molecule names and save as a pickle file. This comes in handy when performing name resolution (many molecules are represented with an english name as opposed to a smiles string). A molecule is understood as having an english name (as opposed to a SMILES string) if it is unresolvable by RDKit.
    6) Merge all the pickled lists of molecule names to create a list of unique molecule names (in merged_molecules_file eg "data/ORD/all_molecule_names.pkl").

    Output:

    1) A pickle file with the cleaned data for each folder of uspto data. NB: Temp always in C, time always in hours
    2) A list of all unique molecule names (in merged_molecules_file)
    """

    if solvents_path == "default":
        solvents_path = None
    else:
        solvents_path = pathlib.Path(solvents_path)

    if name_contains_substring == "":
        name_contains_substring = None

    main(
        data_path=pathlib.Path(data_path),
        ord_file_ending=ord_file_ending,
        trust_labelling=trust_labelling,
        output_path=pathlib.Path(output_path),
        pickled_data_folder=pickled_data_folder,
        solvents_path=solvents_path,
        molecule_names_folder=molecule_names_folder,
        merged_molecules_file=pathlib.Path(merged_molecules_file),
        use_multiprocessing=use_multiprocessing,
        name_contains_substring=name_contains_substring,
        inverse_substring=inverse_substring,
        overwrite=overwrite,
    )


def main(
    data_path: pathlib.Path,
    ord_file_ending: str,
    trust_labelling: bool,
    output_path: pathlib.Path,
    pickled_data_folder: str,
    solvents_path: typing.Optional[pathlib.Path],
    molecule_names_folder: str,
    merged_molecules_file: pathlib.Path,
    use_multiprocessing: bool,
    name_contains_substring: typing.Optional[str],
    inverse_substring: bool,
    overwrite: bool,
):
    """
    After downloading the USPTO dataset from ORD, this script will extract the data and write it to pickle files.
        Example:


    Args:

    1) data_path: pathlib.Path
        - Path to the folder that contains the ORD data
    2) ord_file_ending: str
        - the file ending of the ord data typically ".pb.gz"
    3) trust_labelling: Bool
        - If True, maintain the labelling and ordering of the original data.
        - If False: Trust the mapped reaction more than the labelled data. A reaction string should be of the form reactants>agents>products; however, agents (particularly reagents) may sometimes appear as reactants on the LHS, so any molecules on the LHS we re-label as a reagent if it (i) does not contain any atom mapped atoms, (ii) the molecule appears on both the LHS and RHS (ie it is unreacted). Note that the original labelling is trusted if the reaction is not mapped. The agents list consists of catalysts, reagents and solvents; any molecules that occur in the set of solvents are extracted from the agents list and re-labelled as solvents, while the remaining molecules remain labelled as agents. Then the list of agents and solvents is sorted alphabetically, and finally any molecules that contain a metal were moved to the front of the agents list; ideally these lists be sorted by using chemical reasoning (e.g. by amount or importance), however this data doesn't exist, so we sort alphabetically and move metal containing molecules to the front (since its likely to be a catalyst) to at least add some order, albeit an arbitrary one. Prior work indicates that sequential prediction of conditions outperforms predicting all conditions in a single output layer (https://doi.org/10.1021/acscentsci.8b00357), so ordering may be helpful.
    4) output_path: pathlib.Path
        - The path to the folder than will contain the pickled_data_folder and molecule_names_folder
    5) pickled_data_folder: str
        - The name of folder than contains the pickle data structures
    6) solvents_path: typing.Optional[pathlib.Path]
        - The path to the solvents csv, if None will use the default
    7) molecule_names_folder: str
        - The name of the folder that contains the molecule_name pickles per folder
    8) merged_molecules_file: pathlib.Path
        - The path to the merged_molecules pickle file
    9) use_multiprocessing: bool
        - Boolean to make the processing of each ORD file done with multiprocessing
    10) name_contains_substring: typing.Optional[str]
        - A filter for the ORD file names, will only extract files that includes the str. If left empty will not search for anything. For example 'uspto' grabs only uspto data
    11) inverse_substring: bool
        - Inversed the name contains substring, so name_contains_substring='uspto' & inverse_substring=True will exclude names with uspto in
    12) overwrite: bool
        - If true, will overwrite existing files, else will through an error if a file exists.


    Functionality:

    1) USPTO data extracted from ORD comes in a large number of files (.pd.gz) batched in a large number of sub-folders. First step is to extract all filepaths that contain USPTO data (by checking whether the string 'uspto' is contained in the filename).
    2) Iterated over all filepaths to extract the following data:
        - The mapped reaction (unchanged)
        - Reactants and products (extracted from the mapped reaction)
        - Reagents: some reagents were extracted from the mapped reaction (if between the >> symbols) while other reagents were labelled as reagents in ORD
        - Solvents and catalysts: labelled as such in ORD
        - Temperature: All temperatures were converted to Celcius. If only the control type was specified, the following mapping was used: 'AMBIENT' -> 25, 'ICE_BATH' -> 0, 'DRY_ICE' -> -78.5, 'LIQUID_NITROGEN' -> -196.
        - Time: All times were converted to hours.
        - Yield (for each product): The PERCENTAGEYIELD was preferred, but if this was not available, the CALCULATEDPERCENTYIELD was used instead. If neither was available, the value was set to np.nan.
        - Procedure_details: Plain text string describing the procedure in English.
    3) Canonicalisation and light cleaning
        - All SMILES strings were canonicalised using RDKit.
        - A "replacements dictionary" was created to replace common names with their corresponding SMILES strings. This dictionary was created by iterating over the most common names in the dataset and replacing them with their corresponding SMILES strings. This was done semi-manually, with some dictionary entries coming from solvents.csv and others being added within the script (in the build_replacements function; mainly concerning catalysts).
        - The final light cleaning step depends on the value of trust_labelling (see above, in the Args section).
        - Reactions will only be added if the reactants and products are different (i.e. no crystalisation reactions etc.)
    4) Build a pandas DataFrame from this data (one for each ORD file), and save each as a pickle file
    5) Create a list of all molecule names and save as a pickle file. This comes in handy when performing name resolution (many molecules are represented with an english name as opposed to a smiles string). A molecule is understood as having an english name (as opposed to a SMILES string) if it is unresolvable by RDKit.
    6) Merge all the pickled lists of molecule names to create a list of unique molecule names (in merged_molecules_file eg "data/ORD/all_molecule_names.pkl").

    Output:

    1) A pickle file with the cleaned data for each folder of uspto data. NB: Temp always in C, time always in hours
    2) A list of all unique molecule names (in merged_molecules_file)
    """

    LOG.info("starting extraction")
    start_time = datetime.datetime.now()

    pickled_data_path = output_path / pickled_data_folder
    molecule_name_path = output_path / molecule_names_folder

    pickled_data_path.mkdir(parents=True, exist_ok=True)
    molecule_name_path.mkdir(parents=True, exist_ok=True)

    files = get_file_names(directory=data_path, file_ending=ord_file_ending)

    solvents_set = orderly.data.get_solvents_set(path=solvents_path)
    manual_replacements_dict = get_manual_replacements_dict(solvents_path=solvents_path)

    kwargs = {
        "output_path": output_path,
        "trust_labelling": trust_labelling,
        "manual_replacements_dict": manual_replacements_dict,
        "solvents_set": solvents_set,
        "pickled_data_folder": pickled_data_folder,
        "molecule_names_folder": molecule_names_folder,
        "name_contains_substring": name_contains_substring,
        "inverse_substring": inverse_substring,
        "overwrite": overwrite,
    }

    if use_multiprocessing:
        # somewhat dangerous imports so keeping localised
        import multiprocessing
        import joblib

        num_cores = multiprocessing.cpu_count()
        with tqdm.contrib.logging.logging_redirect_tqdm(loggers=[LOG]):
            joblib.Parallel(n_jobs=num_cores)(
                joblib.delayed(extract)(file=file, **kwargs)
                for file in tqdm.tqdm(files)
            )
    else:
        with tqdm.contrib.logging.logging_redirect_tqdm(loggers=[LOG]):
            for file in tqdm.tqdm(files):
                extract(file=file, **kwargs)

    merge_pickled_mol_names(
        molecule_names_path=molecule_name_path,
        output_file_path=output_path / merged_molecules_file,
        overwrite=overwrite,
        molecule_names_file_ending=".pkl",
    )
    end_time = datetime.datetime.now()
    LOG.info("Duration: {}".format(end_time - start_time))
