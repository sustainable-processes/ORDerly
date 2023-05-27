import dataclasses
import datetime
import json
import logging
import os
import pathlib
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
import tqdm
import tqdm.contrib.logging
from click_loglevel import LogLevel
from rdkit import Chem as rdkit_Chem
from rdkit.rdBase import BlockLogs as rdkit_BlockLogs

from orderly.types import *

LOG = logging.getLogger(__name__)

import orderly.data.util


@dataclasses.dataclass(kw_only=True)
class Cleaner:
    """Loads in the extracted data and removes invalid/undesired reactions.
    1) Merge the parquet files generated during orderly.extract into a df
    2) Remove reactions without any products and/or reactants (remove_reactions_with_no_reactants, remove_reactions_with_no_products, remove_reactions_with_no_conditions)
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
        remove_reactions_with_no_conditions (bool): Remove reactions with no conditions (e.g. no solvent, catalyst, reagent, agent)
        remove_reactions_with_no_solvents (bool): Remove reactions with no solvents
        remove_reactions_with_no_agents (bool): Remove reactions with no agents
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
    remove_reactions_with_no_conditions: bool
    remove_reactions_with_no_solvents: bool
    remove_reactions_with_no_agents: bool
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
    set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn: bool
    remove_rxn_with_unresolved_names: bool
    set_unresolved_names_to_none: bool
    drop_duplicates: bool
    scramble: bool
    disable_tqdm: bool

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
        for file in sorted(self.ord_extraction_path.glob("*.parquet")):
            LOG.debug(f"Reading {file=}")
            extracted_df = pd.read_parquet(file)
            dfs.append(extracted_df)
            LOG.debug(f"Read {file=}")
        LOG.info("Successfully read all data")
        df = pd.concat(dfs, ignore_index=True)

        # the columns below have an unstandardised length so we fill the nan values with a specific string to avoid ambiguity of none / nan
        target_strings = (
            "agent",
            "solvent",
            "reagent",
            "catalyst",
            "product",
            "reactant",
        )
        target_columns = self._get_columns_beginning_with_str(
            columns=df.columns,
            target_strings=target_strings,
        )
        df[target_columns] = df[target_columns].fillna("<missing>")
        df[target_columns] = df[target_columns].astype("string")

        LOG.debug("Unifying missing data to None")
        df = df.replace("<missing>", None)
        LOG.debug("Unified missing data to None")
        return df

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
        df: pd.DataFrame,
        component_name: str,
        number_of_columns_to_keep: int,
        num_cat_cols_to_keep: int = 1,
        recursive_counter: int = 0,
    ) -> pd.DataFrame:
        LOG.info(
            f"Removing reactions with too many components for {component_name=} threshold={number_of_columns_to_keep}"
        )
        if recursive_counter > 5:
            raise ValueError(
                "Too many recursive calls attempting to remap catalyst to reagent columns"
            )

        if number_of_columns_to_keep == -1:
            df = df.sort_index(axis=1)
            df = df.reset_index(drop=True)
            return df

        # Filter the columns that start with component_name
        component_columns = [
            col for col in df.columns if col.startswith(component_name)
        ]
        # If there are more columns than the threshold, remove the excess ones
        if len(component_columns) > number_of_columns_to_keep:
            columns_to_remove = component_columns[number_of_columns_to_keep:]

            # Create a boolean mask for rows with missing values in the columns to remove
            mask = df[columns_to_remove].isnull().all(axis=1)

            # Filter the DataFrame using the mask and drop the columns to remove
            df = df.loc[mask].drop(columns_to_remove, axis=1)

        # And if there are fewer than expected, add a column of Nones
        elif len(component_columns) < number_of_columns_to_keep:
            LOG.warning(
                f"There are only {len(component_columns)} {component_name} columns, but {number_of_columns_to_keep} were requested. Adding empty columns (or replacing reagent with catalyst)."
            )

            # Need to check that there's enough catalyst columns to replace the reagent columns, if not we'll just add empty columns
            # If there are enough catalyst columns
            #   then, rename some of the catalyst columns as reagent columns
            cat_cols = [col for col in df.columns if col.startswith("catalyst")]
            if (
                (component_name == "reagent")
                and (len(component_columns) == 0)
                and (len(cat_cols) - num_cat_cols_to_keep > 0)
            ):
                # TODO: add catch for when the number of reagent columns is not 0? Mix reagent and catalyts columns?

                LOG.warning(f"Replacing reagent with catalyst")
                cols_to_rename = cat_cols[number_of_columns_to_keep:]
                for i, col in enumerate(cols_to_rename):
                    df[f"reagent_{i:03d}"] = df[col]
                    df = df.drop(columns=col)
                df = Cleaner._remove_reactions_with_too_many_of_component(
                    df=df,
                    component_name="reagent",
                    number_of_columns_to_keep=number_of_columns_to_keep,
                    num_cat_cols_to_keep=num_cat_cols_to_keep,
                    recursive_counter=recursive_counter + 1,
                )

            else:
                num_columns_to_add = number_of_columns_to_keep - len(component_columns)
                column_names_to_add = [
                    f"{component_name}_{i:03d}"
                    for i in range(
                        len(component_columns),
                        len(component_columns) + num_columns_to_add,
                    )
                ]
                for new_col_name in column_names_to_add:
                    empty_col = [pd.NA] * df.shape[
                        0
                    ]  # create a column of Nones the same length as the df
                    new_columns = pd.DataFrame(
                        columns=[new_col_name], data=empty_col
                    )  # these columns are all empty
                    df = pd.concat([df, new_columns], axis=1)

        df = df.sort_index(axis=1)
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def _remove_rxn_with_no_conditions(
        df: pd.DataFrame, components: Optional[List[str]] = None
    ) -> pd.DataFrame:
        LOG.info("Removing reactions with no conditions")
        if components is None:
            components = ["catalyst", "solvent", "agent", "reagent"]
        # Check for rows with all None values in the specified columns
        mask = (
            df.loc[:, df.columns.str.startswith(tuple(components))].isna().all(axis=1)
        )

        # Remove rows with all None values
        filtered_df = df[~mask]

        return filtered_df

    @staticmethod
    def _del_rows_empty_in_this_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
        # Helper function to remove reactions with no reactants or products (hence why we're only looking at the first column)
        # Replace 'none' with np.nan in 'products_000' column
        column_name = col + "_000"
        # df[column_name] = df[column_name].replace({None: np.nan})
        # Get indices where col is NaN
        nan_indices = df.index[df[column_name].isna()]

        # Create a mask for all columns that start with 'products_'
        mask = df.columns.str.startswith(col)

        # For all indices where 'products_000' is NaN, check if any column starting with 'products_'
        # contains a non-null value
        for index in nan_indices:
            if not df.loc[index, mask].isna().all():
                raise ValueError(
                    f"Non-null value found in {col} columns for index {index} despite {column_name} being null"
                )

        # Remove rows from df using the mask
        df = df.drop(nan_indices)

        LOG.info(f"Removing rows with empty {col}")
        df = df.dropna(subset=[column_name])
        return df

    @staticmethod
    def _remove_with_inconsistent_yield(
        df: pd.DataFrame, num_product: int
    ) -> pd.DataFrame:
        # Keep rows with yield <= 100 or missing yield values
        mask = pd.Series(data=True, index=df.index)  # start with all rows selected
        for i in range(num_product):
            yield_col = f"yield_{i:03d}"
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
            LOG.info(f"map rare molecules to other for {col=}")
            # Find the values that occur less frequently than the minimum frequency threshold
            rare_values = {
                i: "other"
                for i in value_counts[
                    value_counts < min_frequency_of_occurrence
                ].index.tolist()
            }

            # Map the rare values to 'other'
            df[col] = df[col].map(
                lambda x: rare_values.get(x, x)
            )  # equivalent to series = series.replace(rare_values)
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

    @staticmethod
    def _get_columns_beginning_with_str(
        columns: List[str], target_strings: Optional[Tuple[str, ...]] = None
    ) -> List[str]:
        """goes through the column in a dataframe and adds columns that start with a string in the target strings"""
        if target_strings is None:
            target_strings = (
                "agent",
                "solvent",
                "reagent",
                "catalyst",
                "product",
                "reactant",
            )

        return sorted([col for col in columns if col.startswith(target_strings)])

    def _sort_row(row: pd.Series) -> pd.Series:
        return pd.Series(sorted(row, key=lambda x: pd.isna(x)), index=row.index)  # type: ignore [no-any-return]

    def _sort_row_relative(
        row: pd.Series, to_sort: List[str], to_keep_ordered: List[str]
    ) -> pd.Series:
        target_row = row[to_sort].reset_index(drop=True).sort_values(na_position="last")
        rel_row = row[to_keep_ordered].reset_index(drop=True)
        rel_row = rel_row[target_row.index]
        rel_row.index = to_keep_ordered
        target_row.index = to_sort
        row = pd.concat([target_row, rel_row])
        return row

    @staticmethod
    def _move_none_to_after_data(
        df: pd.DataFrame, target_strings: Tuple[str, ...]
    ) -> pd.DataFrame:
        LOG.info(f"Moving None to after data for {target_strings=}")
        for molecule_type in target_strings:  # i.e. reactant
            ordering_target_columns = Cleaner._get_columns_beginning_with_str(
                columns=df.columns,
                target_strings=(molecule_type,),
            )
            if len(ordering_target_columns) == 0:
                continue
            if molecule_type == "product":
                yield_columns = Cleaner._get_columns_beginning_with_str(
                    columns=df.columns,
                    target_strings=("yield",),
                )
                if len(yield_columns) != len(ordering_target_columns):
                    raise ValueError(
                        f"{len(yield_columns)=} must be the same as {len(ordering_target_columns)=}"
                    )
                df.loc[:, ordering_target_columns + yield_columns] = df.loc[
                    :, ordering_target_columns + yield_columns
                ].apply(
                    lambda x: Cleaner._sort_row_relative(
                        x, ordering_target_columns, yield_columns
                    ),
                    axis=1,
                )
            else:
                # Apply a lambda function to sort the elements within each row, placing None values last
                df.loc[:, ordering_target_columns] = df.loc[
                    :, ordering_target_columns
                ].apply(
                    Cleaner._sort_row,
                    axis=1,
                )

        return df

    @staticmethod
    def _scramble(
        df: pd.DataFrame,
        components: Tuple[str, ...] = ("agent", "solvent", "catalyst", "reagent"),
        seed: int = 42,
    ) -> pd.DataFrame:
        """Scrambles the order of the reactants (ie between reactant_001, reactant_002, etc). Ordering of prodcuts, agents, solvents, reagents, and catalysts will also be scrambled. This is done to prevent the model from learning the order of the molecules, which is not important for the reaction prediction task. It only done at the very end because scrambling can be non-deterministic between versions/operating systems, so it would be difficult to debug if done earlier in the pipeline."""
        list_of_dfs = []
        all_component_cols = []
        np.random.seed(seed)
        for component_name in components:
            component_columns = [
                col for col in df.columns if col.startswith(component_name)
            ]
            all_component_cols += component_columns
            sub_df = df[component_columns]
            if len(sub_df.columns) > 1:
                sub_df = sub_df.apply(
                    lambda row: pd.Series(
                        np.random.permutation(row.values), index=row.index
                    ),
                    axis=1,
                )
            list_of_dfs.append(sub_df)

        shuffled_sub_df = pd.concat(list_of_dfs, axis=1)
        non_target_cols_df = df.drop(all_component_cols, axis=1)
        new_df = pd.concat([non_target_cols_df, shuffled_sub_df], axis=1)
        assert sorted(new_df.columns.tolist()) == sorted(df.columns.tolist())
        return new_df

    @staticmethod
    def _replace_None_with_NA(
        df: pd.DataFrame,
        components: Tuple[str, ...] = ("agent", "solvent", "catalyst", "reagent"),
    ) -> pd.DataFrame:
        """The scrambling or move none to end of list functions populate empty values with None. This function replaces those None values with NA, which is the value used by the model to indicate an empty value."""
        LOG.info("Replacing None with NA")
        all_component_cols = []
        for component_name in components:
            component_columns = [
                col for col in df.columns if col.startswith(component_name)
            ]
            all_component_cols += component_columns

        sub_df = df[all_component_cols]
        sub_df = sub_df.applymap(lambda x: pd.NA if x is None else x)

        df = df.drop(all_component_cols, axis=1)
        df = pd.concat([df, sub_df], axis=1)

        return df

    def _get_dataframe(self) -> pd.DataFrame:
        _ = rdkit_BlockLogs()
        # Merge all the extracted data into one big df

        LOG.info("Getting dataframe from extracted ORDs")
        df = self._merge_extracted_ords()

        LOG.info(f"All data length: {df.shape[0]}")
        LOG.info("Handle unresolvable names")

        LOG.info(
            f"{self.set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=}"
        )
        LOG.info(f"{self.remove_rxn_with_unresolved_names=}")
        LOG.info(f"{self.set_unresolved_names_to_none=}")

        # Handling unresolvable names:
        ### This is not straight forward, because there can be many different reasons for unresolable names
        ### There are two sources of molecules the reaction string, and the ORD reaction.input object; the reaction.input object is more likely to contain unresolvable names (such as '5' or 'Ester' etc.). The two obvious approaches to handling these:
        ### 1. Remove the entire reactions/row with unresolvable names
        ### 2. Set the unresolvable names to None (thus maintaining the reaction/row)
        ### Option 1 may delete reactions that are actually useful, and option 2 may result in keeping reactions that don't make sense because a component is missing.
        ### Our compromise is to set unresolvable names to none when we have a mapped reaction string, and delete the reaction if we don't have a mapped reaction string (since the presence of a mapped rxn string makes the reaction much more trustworthy).
        ### If you don't want to use this compromise, you can also set the unresolvable names to none for all reactions, or delete all reactions with unresolvable names, or retain all the unresolvable names (by setting all 3 bools to False).

        target_strings = (
            "agent",
            "solvent",
            "reagent",
            "catalyst",
            "product",
            "reactant",
        )

        target_columns = self._get_columns_beginning_with_str(
            columns=df.columns,
            target_strings=target_strings,
        )

        if self.set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn:
            LOG.info(
                f"Before removing reactions without mapped rxn that also have unresolvable names: {df.shape[0]}"
            )
            mask_is_mapped = df["is_mapped"]
            LOG.info("Got mask for if reactions are mapped")
            mapped_rxn_df = df.loc[mask_is_mapped]
            not_mapped_rxn_df = df.loc[~mask_is_mapped]

            LOG.info(
                f"Set unresolved names to none for {target_columns}: {df.shape[0]}"
            )
            mapped_rxn_dict_with_replacements = {}
            # set unresolved names to <unresolved>
            mtr = {i: None for i in self.molecules_to_remove}
            for col in target_columns:
                LOG.info(f"Applying nones to {col=}")
                mapped_rxn_dict_with_replacements[col] = mapped_rxn_df[col].map(
                    lambda x: mtr.get(x, x)
                )  # equivalent to series = series.replace(self.molecules_to_remove, <unresolved>)
            mapped_rxn_df_with_replacements = pd.DataFrame(
                mapped_rxn_dict_with_replacements
            )
            # Add back the non-target columns to the df
            mapped_rxn_df_with_replacements = pd.concat(
                [
                    mapped_rxn_df_with_replacements,
                    mapped_rxn_df.loc[:, ~mapped_rxn_df.columns.isin(target_columns)],
                ],
                axis=1,
            )

            # remove reactions with unresolved names
            if not not_mapped_rxn_df.empty:
                for col in tqdm.tqdm(
                    target_columns,
                    disable=self.disable_tqdm,
                ):
                    LOG.info(f"Attempting to remove reactions for {col}")
                    not_mapped_rxn_df_with_del_rows = not_mapped_rxn_df[
                        ~not_mapped_rxn_df[col].isin(self.molecules_to_remove)
                    ]
                    LOG.info(
                        f"Removed reactions with unresolved names for {col}: {df.shape[0]}"
                    )

                # concat the dfs again
                df = pd.concat(
                    [mapped_rxn_df_with_replacements, not_mapped_rxn_df_with_del_rows]
                )
            else:
                df = mapped_rxn_df_with_replacements

            LOG.info(
                f"After removing reactions without mapped rxn that also have unresolvable names: {df.shape[0]}"
            )

        elif self.remove_rxn_with_unresolved_names:
            LOG.info(
                f"Before removing reactions with unresolvable names: {df.shape[0]}"
            )
            for col in tqdm.tqdm(target_columns, disable=self.disable_tqdm):
                df = df[~df[col].isin(self.molecules_to_remove)]
            LOG.info(f"After removing reactions with unresolvable names: {df.shape[0]}")

        elif self.set_unresolved_names_to_none:
            LOG.info(
                f"Setting unresolvable names to None (without removing any reactions)"
            )
            LOG.info(
                f"Set unresolved names to none for {target_columns}: {df.shape[0]}"
            )
            dict_with_replacements = {}
            # set unresolved names to <unresolved>
            mtr = {i: None for i in self.molecules_to_remove}
            for col in target_columns:
                LOG.info(f"Applying nones to {col=}")
                dict_with_replacements[col] = df[col].map(
                    lambda x: mtr.get(x, x)
                )  # equivalent to series = series.replace(self.molecules_to_remove, <unresolved>)
            df_with_replacements = pd.DataFrame(dict_with_replacements)
            # Add back the non-target columns to the df
            df = pd.concat(
                [df_with_replacements, df.loc[:, ~df.columns.isin(target_columns)]],
                axis=1,
            )

        # Remove reactions with too many of a certain component
        num_cat_cols_to_keep = self._get_number_of_columns_to_keep()["catalyst"]
        for col in [
            "reactant",
            "product",
            "yield",
            "solvent",
            "agent",
            "reagent",
            "catalyst",
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
                num_cat_cols_to_keep=num_cat_cols_to_keep,
            )
            LOG.info(f"After removing reactions with too many {col}s: {df.shape[0]}")

            df = Cleaner._del_rows_empty_in_this_col(df, "product")

        # Remove reactions with no reactants
        if self.remove_reactions_with_no_reactants:
            LOG.info(f"Before removing reactions with no reactants: {df.shape[0]}")
            df = Cleaner._del_rows_empty_in_this_col(df, "reactant")
            LOG.info(f"After removing reactions with no reactants: {df.shape[0]}")
        # Remove reactions with no products
        if self.remove_reactions_with_no_products:
            LOG.info(f"Before removing reactions with no products: {df.shape[0]}")
            df = Cleaner._del_rows_empty_in_this_col(df, "product")
            LOG.info(f"After removing reactions with no products: {df.shape[0]}")
        if self.remove_reactions_with_no_solvents:
            LOG.info(f"Before removing reactions with no solvents: {df.shape[0]}")
            df = Cleaner._del_rows_empty_in_this_col(df, "solvent")
            LOG.info(f"After removing reactions with no solvents: {df.shape[0]}")
        if self.remove_reactions_with_no_agents:
            if "agent_000" in df.columns:
                LOG.info(f"Before removing reactions with no agents: {df.shape[0]}")
                df = Cleaner._del_rows_empty_in_this_col(df, "agent")
                LOG.info(f"After removing reactions with no agents: {df.shape[0]}")
            else:
                LOG.info(
                    f"Before removing reactions with no reagents AND no catalysts: {df.shape[0]}"
                )
                df = Cleaner._remove_rxn_with_no_conditions(
                    df, components=["catalyst", "reagent"]
                )
                LOG.info(
                    f"After removing reactions with no reagents AND no catalysts: {df.shape[0]}"
                )

        if self.remove_reactions_with_no_conditions:
            LOG.info(
                f"Before removing reactions with no conditions (ie no solvents AND no agents): {df.shape[0]}"
            )
            df = Cleaner._remove_rxn_with_no_conditions(
                df, components=["catalyst", "solvent", "agent", "reagent"]
            )
            LOG.info(
                f"After removing reactions with no conditions (ie no solvents AND no agents): {df.shape[0]}"
            )

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

            columns_to_count_from = self._get_columns_beginning_with_str(
                columns=df.columns,
                target_strings=("agent", "solvent", "reagent", "catalyst"),
            )
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

        # drop duplicates deals with any final duplicates from mapping rares to other
        if self.drop_duplicates:
            LOG.info(f"Before removing duplicates: {df.shape[0]}")
            df = df.drop_duplicates()
            LOG.info(f"After removing duplicates: {df.shape[0]}")

        df.reset_index(inplace=True, drop=True)

        if self.scramble:
            LOG.info(f"Scrambling the order of the components")
            components = ("agent", "solvent", "reagent", "catalyst")
            df = Cleaner._scramble(df, components)
            df = Cleaner._move_none_to_after_data(df, components)
            df = Cleaner._replace_None_with_NA(df, components)
        df = df.sort_index(axis=1)
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
    "--remove_reactions_with_no_reactants",
    type=bool,
    default=True,
    show_default=True,
    help="Remove reactions with no reactants",
)
@click.option(
    "--remove_reactions_with_no_products",
    type=bool,
    default=True,
    show_default=True,
    help="Remove reactions with no products",
)
@click.option(
    "--remove_reactions_with_no_conditions",
    type=bool,
    default=False,
    show_default=True,
    help="Remove reactions with no rxn conditions (i.e. no solvents AND no agents. This is different to remove_reactions_with_no_solvents=True and remove_reactions_with_no_agents=True since this will remove reactions with no solvents OR no agents )",
)
@click.option(
    "--remove_reactions_with_no_solvents",
    type=bool,
    default=True,
    show_default=True,
    help="Remove reactions with no solvents",
)
@click.option(
    "--remove_reactions_with_no_agents",
    type=bool,
    default=True,
    show_default=True,
    help="Remove reactions with no agents (ie no reagents AND no catalysts). Does not consider solvents",
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
    "--drop_duplicates",
    type=bool,
    default=True,
)
@click.option(
    "--scramble",
    type=bool,
    default=False,
    help="If True, the order of the reactants be scrambled (ie between reactant_001, reactant_002, etc). Ordering of prodcuts, agents, solvents, reagents, and catalysts will also be scrambled. Will also scramble the reaction indices. This is done to prevent the model from learning the order of the molecules, which is not important for the reaction prediction task. It only done at the very end because scrambling can be non-deterministic between versions/operating systems, so it would be difficult to debug if done earlier in the pipeline.",
)
@click.option(
    "--train_test_split_fration",
    type=float,
    default=0.9,
    help="If True, applies random split to create train and test set (90/10); a dict of the train and test indices will be saved to the output_path (instead of a df)",
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
    remove_reactions_with_no_conditions: bool,
    remove_reactions_with_no_solvents: bool,
    remove_reactions_with_no_agents: bool,
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
    drop_duplicates: bool,
    scramble: bool,
    train_test_split_fration: float,
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
    file_name = pathlib.Path(output_path).name
    if file_name.endswith(".parquet"):
        file_name = file_name[: -len(".parquet")]
    _log_file = pathlib.Path(output_path).parent / f"{file_name}_clean.log"
    if log_file != "default_path_clean.log":
        _log_file = pathlib.Path(log_file)

    main(
        output_path=pathlib.Path(output_path),
        ord_extraction_path=pathlib.Path(ord_extraction_path),
        molecules_to_remove_path=pathlib.Path(molecules_to_remove_path),
        consistent_yield=consistent_yield,
        remove_reactions_with_no_reactants=remove_reactions_with_no_reactants,
        remove_reactions_with_no_products=remove_reactions_with_no_products,
        remove_reactions_with_no_conditions=remove_reactions_with_no_conditions,
        remove_reactions_with_no_solvents=remove_reactions_with_no_solvents,
        remove_reactions_with_no_agents=remove_reactions_with_no_agents,
        num_reactant=num_reactant,
        num_product=num_product,
        num_solv=num_solv,
        num_agent=num_agent,
        num_cat=num_cat,
        num_reag=num_reag,
        min_frequency_of_occurrence=min_frequency_of_occurrence,
        map_rare_molecules_to_other=map_rare_molecules_to_other,
        set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn,
        remove_rxn_with_unresolved_names=remove_rxn_with_unresolved_names,
        set_unresolved_names_to_none=set_unresolved_names_to_none,
        drop_duplicates=drop_duplicates,
        scramble=scramble,
        train_test_split_fration=train_test_split_fration,
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
    remove_reactions_with_no_conditions: bool,
    remove_reactions_with_no_solvents: bool,
    remove_reactions_with_no_agents: bool,
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
    scramble: bool,
    train_test_split_fration: float,
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
        "remove_reactions_with_no_conditions": remove_reactions_with_no_conditions,
        "remove_reactions_with_no_solvents": remove_reactions_with_no_solvents,
        "remove_reactions_with_no_agents": remove_reactions_with_no_agents,
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
        "drop_duplicates": drop_duplicates,
        "scramble": scramble,
        "train_test_split_fration": train_test_split_fration,
    }

    file_name = pathlib.Path(output_path).name
    if file_name.endswith(".parquet"):
        file_name = file_name[: -len(".parquet")]
    clean_config_path = (
        pathlib.Path(output_path).parent / f"{file_name}_clean_config.json"
    )
    if clean_config_path != "clean.json":
        clean_config_path = pathlib.Path(clean_config_path)
    else:
        clean_config_path = pathlib.Path(output_path).parent / "clean_config.json"
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
        remove_reactions_with_no_reactants=remove_reactions_with_no_reactants,
        remove_reactions_with_no_products=remove_reactions_with_no_products,
        remove_reactions_with_no_conditions=remove_reactions_with_no_conditions,
        remove_reactions_with_no_solvents=remove_reactions_with_no_solvents,
        remove_reactions_with_no_agents=remove_reactions_with_no_agents,
        consistent_yield=consistent_yield,
        num_reactant=num_reactant,
        num_product=num_product,
        num_solv=num_solv,
        num_agent=num_agent,
        num_cat=num_cat,
        num_reag=num_reag,
        min_frequency_of_occurrence=min_frequency_of_occurrence,
        map_rare_molecules_to_other=map_rare_molecules_to_other,
        set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn=set_unresolved_names_to_none_if_mapped_rxn_str_exists_else_del_rxn,
        remove_rxn_with_unresolved_names=remove_rxn_with_unresolved_names,
        set_unresolved_names_to_none=set_unresolved_names_to_none,
        molecules_to_remove=molecules_to_remove,
        drop_duplicates=drop_duplicates,
        scramble=scramble,
        disable_tqdm=disable_tqdm,
    )

    LOG.info(f"completed cleaning, saving to {output_path}")
    if train_test_split_fration not in [0.0, 1.0]:
        df = instance.cleaned_reactions
        LOG.info("Applying random split")
        # Get indices for train and val
        rng = np.random.default_rng(12345)
        train_test_indices = np.arange(df.shape[0])
        rng.shuffle(train_test_indices)
        train_indices = train_test_indices[
            : int(train_test_indices.shape[0] * train_test_split_fration)
        ]
        test_indices = train_test_indices[
            int(train_test_indices.shape[0] * train_test_split_fration) :
        ]

        input_columns = df.columns[df.columns.str.startswith(("reactant", "product"))]
        train_input_df = df.loc[train_indices, input_columns]
        test_input_df = df.loc[test_indices, input_columns]

        train_set_list = [set(row) for _, row in train_input_df.iterrows()]
        test_set_list = [set(row) for _, row in test_input_df.iterrows()]

        matching_indices = []  # List to store the indices of matching rows
        LOG.info("Moving rows from test to train if they are in both")
        for values, index in zip(test_set_list, test_indices):
            if values in train_set_list:
                matching_indices.append(index)

        # drop the matching rows from the test set
        test_indices = test_indices[~np.isin(test_indices, matching_indices)]
        # Add the matching rows to the train set
        train_indices = np.append(train_indices, matching_indices)

        percentage_of_test_data_moved_to_train = len(matching_indices) / len(
            test_indices
        )
        LOG.info(f"{percentage_of_test_data_moved_to_train=}")
        if percentage_of_test_data_moved_to_train > 0.1:
            LOG.warning(
                "More than 10% of the test set was moved the training set. This may indicate a non-diverse dataset."
            )
        train_df = df.loc[train_indices]
        test_df = df.loc[test_indices]

        train_df.to_parquet(output_path.parent / f"{file_name}_train.parquet")
        test_df.to_parquet(output_path.parent / f"{file_name}_test.parquet")
        LOG.info("Saved split data")
    else:
        instance.cleaned_reactions.to_parquet(output_path)
        LOG.info("Saved unsplit data")

    end_time = datetime.datetime.now()
    LOG.info("Cleaning complete, duration: {}".format(end_time - start_time))
