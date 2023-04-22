import logging
import os
import json
from typing import List, Dict
import dataclasses
import datetime
import pathlib
import click

from click_loglevel import LogLevel

import tqdm
import tqdm.contrib.logging
import pandas as pd
import numpy as np
from rdkit import Chem as rdkit_Chem

LOG = logging.getLogger(__name__)

import orderly.data.util


@dataclasses.dataclass(kw_only=True)
class Cleaner:
    """Loads in the extracted data and removes invalid/undesired reactions.
    1) Merge the parquet files generated during orderly.extract into a df
    2) Remove reactions without any products and/or reactants (remove_reactions_with_no_reactants, remove_reactions_with_no_products)
    3) Remove reactions with too many reactants, products, sovlents, agents, catalysts, and reagents (num_reactant, num_product, num_solv, num_agent, num_cat, num_reag)
    4) Remove reactions with inconsistent yields (consistent_yield)
    5) Handle rare molecules (frequency of occurrence < min_frequency_of_occurrence)
        a) If map_rare_molecules_to_other is True, map rare molecules to 'other'
        b) If map_rare_molecules_to_other is False, remove reactions that contain rare molecules
    6) Remove reactions that have a molecule represented by an unresolvable name. This is often an english name or a number.
    7) Remove duplicate reactions
    8) Save the final df

    Output:

    1) A parquet file containing the cleaned data

    Args:
        remove_reactions_with_no_reactants (bool): Remove reactions with no reactants
        remove_reactions_with_no_products (bool): Remove reactions with no products
        consistent_yield (bool): Remove reactions with inconsistent reported yields (e.g. if the sum is under 0% or above 100%. Reactions with nan yields are not removed)
        num_reactant (int): The number of molecules of that type to keep. Keep in mind that if trust_labelling=True in orderly.extract, there will only be agents, but no catalysts/reagents, and if trust_labelling=False, there will only be catalysts and reagents, but no agents. Agents should be seen as a 'parent' category of reagents and catalysts; solvents should fall under this category as well, but since the space of solvents is more well defined (and we have a list of the most industrially relevant solvents which we can refer to), we can separate out the solvents. Therefore, if trust_labelling=True, num_catalyst and num_reagent should be set to 0, and if trust_labelling=False, num_agent should be set to 0. It is recommended to set trust_labelling=True, as we don't believe that the original labelling of catalysts and reagents that reliable; furthermore, what constitutes a catalyst and what constitutes a reagent is not always clear, adding further ambiguity to the labelling, so it's probably best to merge these.
        num_product (int): See help for num_reactant
        num_solv (int): See help for num_reactant
        num_agent (int): See help for num_reactant
        num_cat (int): See help for num_reactant
        num_reag (int): See help for num_reactant

        min_frequency_of_occurrence (int): The minimum number of times a molecule must appear in the dataset (cumulatively, across agents, reagents, solvents, catalysts) to be kept. Infrequently occuring molecules will probably add more noise than signal to the dataset, so it is probably best to remove them (or map to 'other').
        map_rare_molecules_to_other (bool): Will map rare molecules (see above) to the string 'other' rather than removing the reactions with rare molecules
        molecules_to_remove (list[str]: Remove reactions that are represented by a name instead of a SMILES string
        disable_tqdm (bool, optional): Controls the use of tqdm progress bar. Defaults to False.
    """

    ord_extraction_path: pathlib.Path
    remove_reactions_with_no_reactants: bool
    remove_reactions_with_no_products: bool
    consistent_yield: bool
    num_reactant: int
    num_product: int
    num_solv: int
    num_agent: int
    num_cat: int
    num_reag: int
    min_frequency_of_occurrence: int
    map_rare_molecules_to_other: bool
    molecules_to_remove: List[str]

    set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn: bool = True
    remove_rxn_with_unresolved_names: bool = False
    set_unresolved_names_to_none: bool = False

    replace_empty_with_none: bool = True
    drop_duplicates: bool = True
    disable_tqdm: bool = False

    def __post_init__(self) -> None:
        LOG.info("Entered post_init")

        # Only zero or one of the following three bools can be True
        true_count = (
            self.set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn
            + self.remove_rxn_with_unresolved_names
            + self.set_unresolved_names_to_none
        )
        assert true_count <= 1
        self.cleaned_reactions = self._get_dataframe()

    def _merge_extracted_ords(self) -> pd.DataFrame:
        # create one big df of all the extracted data

        LOG.info("Getting merged dataframe from extracted ord files")

        dfs = []
        with tqdm.contrib.logging.logging_redirect_tqdm(loggers=[LOG]):
            for file in tqdm.tqdm(
                self.ord_extraction_path.glob("*.parquet"), disable=self.disable_tqdm
            ):
                LOG.debug(f"Reading {file=}")
                extracted_df = pd.read_parquet(file)
                dfs.append(extracted_df)
                LOG.debug(f"Read {file=}")
        LOG.info("Successfully read all data")
        return pd.concat(dfs, ignore_index=True)

    def _get_number_of_columns_to_keep(self) -> Dict[str, int]:
        return {
            "reactant": self.num_reactant,
            "product": self.num_product,
            "yield": self.num_product,
            "solvent": self.num_solv,
            "agent": self.num_agent,
            "catalyst": self.num_cat,
            "reagent": self.num_reag,
        }

    @staticmethod
    def _remove_reactions_with_too_many_of_component(
        df: pd.DataFrame, component_name: str, number_of_columns_to_keep: int
    ) -> pd.DataFrame:
        LOG.info(
            f"Removing reactions with too many components for {component_name=} threshold={number_of_columns_to_keep}"
        )

        cols = list(df.columns)
        count = 0
        for col in cols:
            if col.startswith(component_name):
                count += 1
        columns_to_remove = []  # columns to remove
        for i in range(count):
            if i >= number_of_columns_to_keep:
                columns_to_remove.append(f"{component_name}_{i:03d}")
        for col in columns_to_remove:
            # Create a boolean mask for the rows with missing values in col
            mask = pd.isnull(df[col])

            # Create a new DataFrame with the selected rows
            df = df.loc[mask]

        df = df.drop(columns_to_remove, axis=1)
        return df

    @staticmethod
    def _remove_rxn_with_no_reactants(df: pd.DataFrame) -> pd.DataFrame:
        LOG.info(f"Removing reactions with no reactants")
        df = Cleaner._del_rows_empty_in_this_col(df, "reactant")
        return df

    @staticmethod
    def _remove_rxn_with_no_products(df: pd.DataFrame) -> pd.DataFrame:
        LOG.info("Removing reactions with no products")
        df = Cleaner._del_rows_empty_in_this_col(df, "product")
        return df

    @staticmethod
    def _del_rows_empty_in_this_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
        # Replace 'none' with np.nan in 'products_0' column
        df[col + "_0"] = df[col + "_0"].replace(None, np.nan)

        # Get indices where col is NaN
        nan_indices = df.index[df[col + "_0"].isna()]

        # Create a mask for all columns that start with 'products_'
        mask = df.columns.str.startswith(col)

        # For all indices where 'products_0' is NaN, check if any column starting with 'products_'
        # contains a non-null value
        for index in nan_indices:
            if not df.loc[index, mask].isna().all():
                raise ValueError(
                    f"Non-null value found in 'products_' columns for index {index} despite products_0 being null"
                )

        # Remove rows from df using the mask
        df = df.drop(nan_indices)

        LOG.info(f"Removing rows with empty {col}")
        df = df.dropna(subset=[col])
        return df

    @staticmethod
    def _remove_with_inconsistent_yield(
        df: pd.DataFrame, num_product: int
    ) -> pd.DataFrame:
        # Keep rows with yield <= 100 or missing yield values
        mask = pd.Series(data=True, index=df.index)  # start with all rows selected
        for i in range(num_product):
            yield_col = "yield_" + str(i)
            yield_mask = (df[yield_col] >= 0) & (df[yield_col] <= 100) | pd.isna(
                df[yield_col]
            )
            mask &= yield_mask

        df = df[mask]

        # sum of yields should be between 0 and 100
        yield_columns = df.filter(like="yield").columns

        # Compute the sum of the yield columns for each row
        df = df.assign(total_yield=df[yield_columns].sum(axis=1))

        # Filter out reactions where the total_yield is less than or equal to 100, or is NaN or None
        mask = (
            (df["total_yield"] <= 100)
            | pd.isna(df["total_yield"])
            | pd.isnull(df["total_yield"])
        )
        df = df[mask]

        # Drop the 'total_yield' column from the DataFrame
        df = df.drop("total_yield", axis=1)
        return df

    @staticmethod
    def _get_value_counts(
        df: pd.DataFrame, columns_to_count_from: List[str]
    ) -> pd.Series:
        """
        Get cumulative value across all columns in columns_to_count_from
        """

        LOG.info(f"Getting value counts for {columns_to_count_from=}")
        # Initialize a list to store the results
        results = []

        # Loop through the columns
        for col in columns_to_count_from:
            # Get the value counts for the column
            results += [df[col].value_counts()]

        total_value_counts = (
            pd.concat(results, axis=0, sort=True).groupby(level=0).sum()
        )
        total_value_counts = total_value_counts.sort_values(ascending=False)
        return total_value_counts

    @staticmethod
    def _map_rare_molecules_to_other(
        df: pd.DataFrame,
        columns_to_transform: List[str],
        value_counts: pd.Series,
        min_frequency_of_occurrence: int,
    ) -> pd.DataFrame:
        """
        Maps rare values in specified columns to 'other'.
        """
        LOG.info(
            f"Mapping rare molecules to 'other' for {columns_to_transform=} with {min_frequency_of_occurrence=}"
        )
        for col in columns_to_transform:
            # Find the values that occur less frequently than the minimum frequency threshold
            rare_values = value_counts[
                value_counts < min_frequency_of_occurrence
            ].index.tolist()
            # Map the rare values to 'other'
            df[col] = df[col].apply(lambda x: "other" if x in rare_values else x)
        return df

    @staticmethod
    def _remove_rare_molecules(
        df: pd.DataFrame,
        columns_to_transform: List[str],
        value_counts: pd.Series,
        min_frequency_of_occurrence: int,
    ) -> pd.DataFrame:
        """
        Removes rows with rare values in specified columns.
        """
        LOG.info(
            f"Removing rare molecules for {columns_to_transform=} with {min_frequency_of_occurrence=}"
        )
        # Get the indices of rows where the column contains a rare value
        rare_values = value_counts[value_counts < min_frequency_of_occurrence].index
        index_union = None

        for col in columns_to_transform:
            mask = df[col].isin(rare_values)
            rare_indices = df.loc[mask].index
            if index_union is None:
                index_union = rare_indices
            else:
                index_union = index_union.union(rare_indices)
        # Remove the rows with rare values
        df = df.drop(index_union)
        return df

    def _get_dataframe(self) -> pd.DataFrame:
        # Merge all the extracted data into one big df

        LOG.info("Getting dataframe from extracted ORDs")
        df = self._merge_extracted_ords()
        LOG.info(f"All data length: {df.shape[0]}")

        # Remove reactions with too many of a certain component
        for col in [
            "reactant",
            "product",
            "yield",
            "solvent",
            "agent",
            "catalyst",
            "reagent",
        ]:
            try:
                number_of_columns_to_keep = self._get_number_of_columns_to_keep()[col]
            except KeyError as exc:
                msg = "KeyError component_name must be one of: reactant, product, yield, solvent, agent, catalyst, reagent"
                LOG.error(msg)
                raise KeyError(msg) from exc

            df = Cleaner._remove_reactions_with_too_many_of_component(
                df,
                component_name=col,
                number_of_columns_to_keep=number_of_columns_to_keep,
            )
            LOG.info(f"After removing reactions with too many {col}s: {df.shape[0]}")

        # Remove reactions with no reactants
        if self.remove_reactions_with_no_reactants:
            LOG.info(f"Before removing reactions with no reactants: {df.shape[0]}")
            df = Cleaner._remove_rxn_with_no_reactants(df)
            LOG.info(f"After removing reactions with no reactant: {df.shape[0]}")

        # Remove reactions with no products
        if self.remove_reactions_with_no_products:
            LOG.info(f"Before removing reactions with no products: {df.shape[0]}")
            df = Cleaner._remove_rxn_with_no_products(df)
            LOG.info(f"After removing reactions with no products: {df.shape[0]}")

        def is_mapped(rxn_str) -> bool:
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

        if self.set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn:
            LOG.info(
                f"Before removing reactions without mapped rxn that also have unresolvable names: {df.shape[0]}"
            )
            mask_is_mapped = df["rxn_str"].apply(is_mapped)
            mapped_rxn_df = df.iloc[mask_is_mapped]
            not_mapped_rxn_df = df.iloc[~mask_is_mapped]

            # set unresolved names to none
            mapped_rxn_df = mapped_rxn_df.replace(self.molecules_to_remove, None)

            # remove reactions with unresolved names
            for col in tqdm.tqdm(not_mapped_rxn_df.columns, disable=self.disable_tqdm):
                not_mapped_rxn_df = not_mapped_rxn_df[
                    ~not_mapped_rxn_df[col].isin(self.molecules_to_remove)
                ]

            # concat the dfs again
            df = pd.concat([mapped_rxn_df, not_mapped_rxn_df])

            LOG.info(
                f"After removing reactions without mapped rxn that also have unresolvable names: {df.shape[0]}"
            )

        elif self.remove_rxn_with_unresolved_names:
            LOG.info(
                f"Before removing reactions without mapped rxn that also have unresolvable names: {df.shape[0]}"
            )
            for col in tqdm.tqdm(df.columns, disable=self.disable_tqdm):
                df = df[~df[col].isin(self.molecules_to_remove)]
            LOG.info(
                f"After removing reactions without mapped rxn that also have unresolvable names: {df.shape[0]}"
            )

        elif self.set_unresolved_names_to_none:
            LOG.info(
                f"Setting unresolvable names to None (without removing any reactions)"
            )
            df = df.replace(self.molecules_to_remove, None)

        # Ensure consistent yield
        if self.consistent_yield:
            LOG.info(
                f"Before removing reactions with inconsistent yields: {df.shape[0]}"
            )
            df = Cleaner._remove_with_inconsistent_yield(
                df, num_product=self.num_product
            )
            LOG.info(
                f"After removing reactions with inconsistent yields: {df.shape[0]}"
            )

        # drop duplicates
        if self.drop_duplicates:
            LOG.info(f"Before removing duplicates: {df.shape[0]}")
            df = df.drop_duplicates()
            LOG.info(f"After removing duplicates: {df.shape[0]}")

        # Remove reactions with rare molecules
        if self.min_frequency_of_occurrence != 0:  # We need to check for rare molecules
            # Define the list of columns to check
            columns_to_count_from = [
                col
                for col in df.columns
                if col.startswith(("agent", "solvent", "reagent", "catalyst"))
            ]
            value_counts = Cleaner._get_value_counts(
                df, columns_to_count_from
            )  # Get the value counts for the subset df[columns_to_check_for_rare_molecules]
            if self.map_rare_molecules_to_other:
                df = Cleaner._map_rare_molecules_to_other(
                    df,
                    columns_to_count_from,
                    value_counts,
                    self.min_frequency_of_occurrence,
                )
            else:
                df = Cleaner._remove_rare_molecules(
                    df,
                    columns_to_count_from,
                    value_counts,
                    self.min_frequency_of_occurrence,
                )
                LOG.info(f"After removing rare molecules: {df.shape[0]}")

        if self.replace_empty_with_none:
            # Replace any instances of an empty string with None
            LOG.info("Replacing empty strings with None")
            df = df.applymap(
                lambda x: None if (isinstance(x, str) and x.strip() == "") else x
            )

        # Replace np.nan with None
        LOG.info("Map np.nan to None")
        df = df.applymap(lambda x: None if pd.isna(x) else x)

        # drop duplicates deals with any final duplicates from mapping rares to other
        if self.drop_duplicates:
            LOG.info(f"Before removing duplicates: {df.shape[0]}")
            df = df.drop_duplicates()
            LOG.info(f"After removing duplicates: {df.shape[0]}")

        df.reset_index(inplace=True, drop=True)
        return df


@click.command()
@click.option(
    "--output_path",
    type=str,
    default="data/orderly/orderly_ord.parquet",
    show_default=True,
    help="The filepath where the cleaned data will be saved",
)
@click.option(
    "--ord_extraction_path",
    default="data/orderly/extracted_ords",
    type=str,
    help="The filepath to the folder than contains the extracted ord data",
)
@click.option(
    "--molecules_to_remove_path",
    default="data/orderly/all_molecule_names.csv",
    type=str,
    help="The path to the file than contains the molecules_names",
)
@click.option(
    "--consistent_yield",
    type=bool,
    default=True,
    show_default=True,
    help="Remove reactions with inconsistent reported yields (e.g. if the sum is under 0% or above 100%. Reactions with nan yields are not removed)",
)
@click.option(
    "--consistent_yield",
    type=bool,
    default=True,
    show_default=True,
    help="Remove reactions with inconsistent reported yields (e.g. if the sum is under 0% or above 100%. Reactions with nan yields are not removed)",
)
@click.option(
    "--consistent_yield",
    type=bool,
    default=True,
    show_default=True,
    help="Remove reactions with inconsistent reported yields (e.g. if the sum is under 0% or above 100%. Reactions with nan yields are not removed)",
)
@click.option(
    "--num_reactant",
    type=int,
    default=5,
    show_default=True,
    help="The number of molecules of that type to keep. Keep in mind that if trust_labelling=True in orderly.extract, there will only be agents, but no catalysts/reagents, and if trust_labelling=False, there will only be catalysts and reagents, but no agents. Agents should be seen as a 'parent' category of reagents and catalysts; solvents should fall under this category as well, but since the space of solvents is more well defined (and we have a list of the most industrially relevant solvents which we can refer to), we can separate out the solvents. Therefore, if trust_labelling=True, num_catalyst and num_reagent should be set to 0, and if trust_labelling=False, num_agent should be set to 0. It is recommended to set trust_labelling=True, as we don't believe that the original labelling of catalysts and reagents that reliable; furthermore, what constitutes a catalyst and what constitutes a reagent is not always clear, adding further ambiguity to the labelling, so it's probably best to merge these.",
)
@click.option(
    "--num_product",
    type=int,
    default=5,
    show_default=True,
    help="See help for num_reactant",
)
@click.option(
    "--num_solv",
    type=int,
    default=2,
    show_default=True,
    help="See help for num_reactant",
)
@click.option(
    "--num_agent",
    type=int,
    default=3,
    show_default=True,
    help="See help for num_reactant",
)
@click.option(
    "--num_cat",
    type=int,
    default=0,
    show_default=True,
    help="See help for num_reactant",
)
@click.option(
    "--num_reag",
    type=int,
    default=0,
    show_default=True,
    help="See help for num_reactant",
)
@click.option(
    "--min_frequency_of_occurrence",
    type=int,
    default=15,
    show_default=True,
    help="The minimum number of times a molecule must appear in the dataset (cumulatively, as an agent, solvent, catalyst, or reagent) to be kept. Infrequently occuring molecules will probably add more noise than signal to the dataset, so it is best to remove them.",
)
@click.option(
    "--map_rare_molecules_to_other",
    type=bool,
    default=True,
    help="If True, molecules that appear less than map_rare_to_other_threshold times will be mapped to the 'other' category. If False, the reaction they appear in will be removed.",
)
@click.option(
    "--set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn",
    type=bool,
    default=True,
)
@click.option(
    "--remove_rxn_with_unresolved_names",
    type=bool,
    default=False,
)
@click.option(
    "--set_unresolved_names_to_none",
    type=bool,
    default=False,
)
@click.option(
    "--replace_empty_with_none",
    type=bool,
    default=True,
)
@click.option(
    "--drop_duplicates",
    type=bool,
    default=True,
)
@click.option("--disable_tqdm", type=bool, default=False, show_default=True)
@click.option(
    "--overwrite",
    type=bool,
    default=False,
    show_default=True,
    help="If true, will overwrite the existing orderly_ord.parquet, else will through an error if a file exists",
)
@click.option(
    "--log_file",
    type=str,
    default="default_path_clean.log",
    show_default=True,
    help="path for the log file for cleaning",
)
@click.option("--log-level", type=LogLevel(), default=logging.INFO)
def main_click(
    output_path: pathlib.Path,
    ord_extraction_path: pathlib.Path,
    molecules_to_remove_path: pathlib.Path,
    remove_reactions_with_no_reactants: bool,
    remove_reactions_with_no_products: bool,
    consistent_yield: bool,
    num_reactant: int,
    num_product: int,
    num_solv: int,
    num_agent: int,
    num_cat: int,
    num_reag: int,
    min_frequency_of_occurrence: int,
    map_rare_molecules_to_other: bool,
    set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn: bool,
    remove_rxn_with_unresolved_names: bool,
    set_unresolved_names_to_none: bool,
    replace_empty_with_none: bool,
    drop_duplicates: bool,
    disable_tqdm: bool,
    overwrite: bool,
    log_file: str,
    log_level: int,
) -> None:
    """
    After running orderly.extract, this script will merge and apply further cleaning to the data.

    Functionality:

    1) Merge the parquet files generated during orderly.extract into a df
    2) Remove reactions with too many reactants, products, sovlents, agents, catalysts, and reagents (num_reactant, num_product, num_solv, num_agent, num_cat, num_reag)
    3) Remove reactions with inconsistent yields (consistent_yield)
    4) Handle rare molecules (frequency of occurrence <= min_frequency_of_occurrence)
        a) If map_rare_molecules_to_other is True, map rare molecules to 'other'
        b) If map_rare_molecules_to_other is False, remove reactions that contain rare molecules
    5) Remove reactions that have a molecule represented by an unresolvable name. This is often an english name or a number.
    6) Remove duplicate reactions
    7) Save the final df

    Output:

    1) A parquet file containing the cleaned data

        NB:
    1) There are lots of places where the code where I use masks to remove rows from a df. These operations could also be done in one line, however, using an operation such as .replace is very slow, and one-liners with dfs can lead to SettingWithCopyWarning. Therefore, I have opted to use masks, which are much faster, and don't give the warning.
    """

    _log_file = pathlib.Path(ord_extraction_path) / ".." / "clean.log"
    if log_file != "default_path_clean.log":
        _log_file = pathlib.Path(log_file)

    main(
        output_path=pathlib.Path(output_path),
        ord_extraction_path=pathlib.Path(ord_extraction_path),
        molecules_to_remove_path=pathlib.Path(molecules_to_remove_path),
        consistent_yield=consistent_yield,
        remove_reactions_with_no_reactants=remove_reactions_with_no_reactants,
        remove_reactions_with_no_products=remove_reactions_with_no_products,
        num_reactant=num_reactant,
        num_product=num_product,
        num_solv=num_solv,
        num_agent=num_agent,
        num_cat=num_cat,
        num_reag=num_reag,
        min_frequency_of_occurrence=min_frequency_of_occurrence,
        map_rare_molecules_to_other=map_rare_molecules_to_other,
        set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn = set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn,
        remove_rxn_with_unresolved_names = remove_rxn_with_unresolved_names,
        set_unresolved_names_to_none = set_unresolved_names_to_none,
        replace_empty_with_none=replace_empty_with_none,
        drop_duplicates=drop_duplicates,
        disable_tqdm=disable_tqdm,
        overwrite=overwrite,
        log_file=_log_file,
        log_level=log_level,
    )


def main(
    output_path: pathlib.Path,
    ord_extraction_path: pathlib.Path,
    molecules_to_remove_path: pathlib.Path,
    consistent_yield: bool,
    remove_reactions_with_no_reactants: bool,
    remove_reactions_with_no_products: bool,
    num_reactant: int,
    num_product: int,
    num_solv: int,
    num_agent: int,
    num_cat: int,
    num_reag: int,
    min_frequency_of_occurrence: int,
    map_rare_molecules_to_other: bool,
    set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn: bool,
    remove_rxn_with_unresolved_names: bool,
    set_unresolved_names_to_none: bool,
    replace_empty_with_none: bool,
    drop_duplicates: bool,
    disable_tqdm: bool,
    overwrite: bool,
    log_file: pathlib.Path = pathlib.Path("cleaning.log"),
    log_level: int = logging.INFO,
) -> None:
    """
    After running orderly.extract, this script will merge and apply further cleaning to the data.

    Functionality:

    1) Merge the parquet files generated during orderly.extract into a df
    2) Remove reactions with too many reactants, products, sovlents, agents, catalysts, and reagents (num_reactant, num_product, num_solv, num_agent, num_cat, num_reag)
    3) Remove reactions with inconsistent yields (consistent_yield)
    4) Handle rare molecules (frequency of occurrence < min_frequency_of_occurrence)
        a) If map_rare_molecules_to_other is True, map rare molecules to 'other'
        b) If map_rare_molecules_to_other is False, remove reactions that contain rare molecules
    5) Remove reactions that have a molecule represented by an unresolvable name. This is often an english name or a number.
    6) Remove duplicate reactions
    7) Save the final df

    Output:

    1) A parquet file containing the cleaned data

        NB:
    1) There are lots of places where the code where I use masks to remove rows from a df. These operations could also be done in one line, however, using an operation such as .replace is very slow, and one-liners with dfs can lead to SettingWithCopyWarning. Therefore, I have opted to use masks, which are much faster, and don't give the warning.
    """

    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        encoding="utf-8",
        format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=log_level,
    )

    if not isinstance(output_path, pathlib.Path):
        e = ValueError(f"Expect pathlib.Path: got {type(output_path)}")
        LOG.error(e)
        raise e
    if not isinstance(ord_extraction_path, pathlib.Path):
        e = ValueError(f"Expect pathlib.Path: got {type(ord_extraction_path)}")
        LOG.error(e)
        raise e
    if not isinstance(molecules_to_remove_path, pathlib.Path):
        e = ValueError(f"Expect pathlib.Path: got {type(molecules_to_remove_path)}")
        LOG.error(e)
        raise e

    if not overwrite:
        if output_path.exists():
            e = FileExistsError(
                "Trying to overwrite the orderly_ord output. Either move the file, change the output_path (output_path) or set to overwrite."
            )
            LOG.error(e)
            raise e

    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = datetime.datetime.now()

    molecules_to_remove = orderly.data.util.load_list(molecules_to_remove_path)

    extract_config_path = ord_extraction_path / ".." / "extract_config.json"

    with open(extract_config_path, "r") as f:
        extract_config = json.load(f)

    if extract_config["trust_labelling"]:
        assert (
            num_agent == 0
        ), "Invalid input: If trust_labelling=True, then num_agent must be 0."
    else:
        assert (num_cat == 0) and (
            num_reag == 0
        ), "Invalid input: If trust_labelling=False in orderly.extract, then num_cat and num_reag must be 0."

    kwargs = {
        "ord_extraction_path": ord_extraction_path,
        "consistent_yield": consistent_yield,
        "remove_reactions_with_no_reactants": remove_reactions_with_no_reactants,
        "remove_reactions_with_no_products": remove_reactions_with_no_products,
        "num_reactant": num_reactant,
        "num_product": num_product,
        "num_solv": num_solv,
        "num_agent": num_agent,
        "num_cat": num_cat,
        "num_reag": num_reag,
        "min_frequency_of_occurrence": min_frequency_of_occurrence,
        "map_rare_molecules_to_other": map_rare_molecules_to_other,
        "set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn": set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn,
        "remove_rxn_with_unresolved_names": remove_rxn_with_unresolved_names,
        "set_unresolved_names_to_none": set_unresolved_names_to_none,
        "replace_empty_with_none": replace_empty_with_none,
        "drop_duplicates": drop_duplicates,
    }

    clean_config_path = output_path.parent / "clean_config.json"
    if not overwrite:
        if clean_config_path.exists():
            e = FileExistsError(
                f"You are trying to overwrite the config file at {clean_config_path} with {overwrite=}"
            )
            LOG.error(e)
            raise e
    copy_kwargs = kwargs.copy()
    copy_kwargs["ord_extraction_path"] = str(copy_kwargs["ord_extraction_path"])
    copy_kwargs["output_path"] = str(output_path)

    with open(clean_config_path, "w") as f:
        json.dump(copy_kwargs, f, indent=4, sort_keys=True)

    LOG.info(f"Beginning extraction for files in {ord_extraction_path}")
    instance = Cleaner(
        ord_extraction_path=ord_extraction_path,
        consistent_yield=consistent_yield,
        remove_reactions_with_no_reactants=remove_reactions_with_no_reactants,
        remove_reactions_with_no_products=remove_reactions_with_no_products,
        num_reactant=num_reactant,
        num_product=num_product,
        num_solv=num_solv,
        num_agent=num_agent,
        num_cat=num_cat,
        num_reag=num_reag,
        min_frequency_of_occurrence=min_frequency_of_occurrence,
        map_rare_molecules_to_other=map_rare_molecules_to_other,
        set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn = set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn,
        remove_rxn_with_unresolved_names = remove_rxn_with_unresolved_names,
        set_unresolved_names_to_none = set_unresolved_names_to_none,
        replace_empty_with_none=replace_empty_with_none,
        drop_duplicates=drop_duplicates,
        disable_tqdm=disable_tqdm,
    )
    LOG.info(f"completed cleaning, saving to {output_path}")
    instance.cleaned_reactions.to_parquet(output_path)
    LOG.info("Saved")

    end_time = datetime.datetime.now()
    LOG.info("Cleaning complete, duration: {}".format(end_time - start_time))
