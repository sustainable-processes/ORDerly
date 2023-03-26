import logging
import os
import typing
import dataclasses
import datetime
import pathlib
import click

import tqdm
import tqdm.contrib.logging
import pandas as pd

LOG = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class Cleaner:
    """

    Args:
        consistent_yield (bool): Remove reactions with inconsistent reported yields (e.g. if the sum is under 0% or above 100%. Reactions with nan yields are not removed)
        num_reactant, num_product, num_solv, num_agent, num_cat, num_reag: (int)

            The number of molecules of that type to keep. Keep in mind that if merge_conditions=True in USPTO_extraction, there will only be agents,
            but no catalysts/reagents, and if merge_conditions=False, there will only be catalysts and reagents, but no agents. Agents should be seen
            as a 'parent' category of reagents and catalysts; solvents should fall under this category as well, but since the space of solvents is
            more well defined (and we have a list of the most industrially relevant solvents which we can refer to), we can separate out the solvents.
            Therefore, if merge_conditions=True, num_catalyst and num_reagent should be set to 0, and if merge_conditions=False, num_agent should be set to
            0. It is recommended to set merge_conditions=True, as we don't believe that the original labelling of catalysts and reagents that reliable;
            furthermore, what constitutes a catalyst and what constitutes a reagent is not always clear, adding further ambiguity to the labelling,
            so it's probably best to merge these.

        num_reactant (int): [description]
        num_product (int): [description]
        num_solv (int): [description]
        num_agent (int): [description]
        num_cat (int): [description]
        num_reag (int): [description]
        min_frequency_of_occurance_primary (int): The minimum number of times a molecule must appear in the dataset to be kept. Infrequently occuring molecules will probably
                                                    add more noise than signal to the dataset, so it is best to remove them. Primary: refers to the first index of columns of
                                                    that type, ie solvent_0, agent_0, catalyst_0, reagent_0
        min_frequency_of_occurance_secondary (int): See above. Secondary: Any other columns than the first.
        include_other_category (bool): Will save reactions with infrequent molecules (below min_frequency_of_occurance_primary/secondary
                                       but above map_rate_to_other) by mapping these molecules to the string 'other'
        map_rate_to_other (bool): Frequency cutoff (see above).
        molecules_to_remove (pd.DataFrame: Remove reactions that are represented by a name instead of a SMILES string
        disable_tqdm (bool, optional): Controls the use of tqdm progress bar. Defaults to False.
    """

    pickles_path: pathlib.Path
    consistent_yield: bool
    num_reactant: int
    num_product: int
    num_solv: int
    num_agent: int
    num_cat: int
    num_reag: int
    min_frequency_of_occurance_primary: int
    min_frequency_of_occurance_secondary: int
    include_other_category: bool
    map_rate_to_other: bool
    molecules_to_remove: pd.DataFrame
    disable_tqdm: bool = False

    def __post_init__(self):
        self.cleaned_reactions = self._get_dataframe()

    def _merge_pickles(self) -> pd.DataFrame:
        # create one big df of all the pickled data

        onlyfiles = [
            f
            for f in os.listdir(self.pickles_path)
            if os.path.isfile(os.path.join(self.pickles_path, f))
        ]
        full_df = pd.DataFrame()

        with tqdm.contrib.logging.logging_redirect_tqdm(loggers=[LOG]):
            for file in tqdm.tqdm(onlyfiles, disable=self.disable_tqdm):
                if file[0] != ".":  # We don't want to try to unpickle .DS_Store
                    filepath = self.pickles_path / file
                    unpickled_df = pd.read_pickle(filepath)
                    full_df = pd.concat([full_df, unpickled_df], ignore_index=True)

        return full_df

    def _get_number_of_columns_to_keep(self) -> typing.Dict[str, int]:
        return {
            "reactant": self.num_reactant,
            "product": self.num_product,
            "yield": self.num_product,
            "solvent": self.num_solv,
            "agent": self.num_agent,
            "catalyst": self.num_cat,
            "reagent": self.num_reag,
        }

    def _remove_reactions_with_too_many_of_component(
        self, df: pd.DataFrame, component_name: str
    ):
        try:
            number_of_columns_to_keep = self.get_number_of_columns_to_keep()[
                component_name
            ]
        except KeyError as exc:
            msg = "component_name must be one of: reactant, product, yield, solvent, agent, catalyst, reagent"
            LOG.error(msg)
            raise KeyError(msg) from exc

        cols = list(df.columns)
        count = 0
        for col in cols:
            if component_name in col:
                count += 1

        columns_to_remove = []  # columns to remove
        for i in range(count):
            if i >= number_of_columns_to_keep:
                columns_to_remove += [component_name + "_" + str(i)]

        for col in columns_to_remove:
            # Create a boolean mask for the rows with missing values in col
            mask = pd.isnull(df[col])

            # Create a new DataFrame with the selected rows
            df = df.loc[mask]

        df = df.drop(columns_to_remove, axis=1)
        return df

    def _remove_rare_molecules(
        self, df: pd.DataFrame, columns: typing.List[str]
    ) -> pd.DataFrame:
        # Molecules that appear keep_as_is_cutoff times or more will be kept as is
        # Molecules that appear less than keep_as_is_cutoff times but more than convert_to_other_cutoff times will be replaced with 'other'
        # Molecules that appear less than convert_to_other_cutoff times will be removed

        # Get the count of each value for all columns
        value_counts = df[columns[0]].value_counts()
        for i in range(1, len(columns)):
            value_counts = value_counts.add(df[columns[i]].value_counts(), fill_value=0)

        for col in columns:
            df = self.filtering_and_removal(df, col, value_counts)
            LOG.info(f"After removing reactions with rare {col}: {len(df)}")

        return df

    def _filtering_and_removal(
        self,
        df: pd.DataFrame,
        col: str,
        value_counts: int,
    ) -> pd.DataFrame:
        if "0" in col:
            upper_cutoff = self.min_frequency_of_occurance_primary
        else:
            upper_cutoff = self.min_frequency_of_occurance_secondary

        pre_len, post_len = 2, 1
        while_loop_counter = 0
        while pre_len > post_len:
            while_loop_counter += 1
            if while_loop_counter > 15:
                exc = TimeoutError(
                    "Looped to many times in trying to remove rare molecules"
                )
                LOG.exception(exc)
                raise exc

            # When we remove rows that feature rare molecules, we iterate through the columns. This means that we may remove a row with a rare molecule with
            # a frequency that was just above the threshold before, and just under the threshold after. So we loop through this code again and again
            # until all solvent and agent molecules appear at least cutoff times.
            pre_len = len(df)

            if self.include_other_category:
                # Select the values where the count is less than cutoff
                set_to_other = value_counts[
                    (value_counts >= self.use_other_label)
                    & (value_counts < upper_cutoff)
                ].index
                set_to_other = set(set_to_other)
                # Create a boolean mask for the rows with values in set_to_other
                mask = df[col].isin(set_to_other)

                # Replace the values in the selected rows and columns with 'other'
                df.loc[mask, col] = "other"

                # Remove rows with a very rare molecule
                to_remove = value_counts[value_counts < self.use_other_label].index

            else:
                # Remove rows with a very rare molecule
                to_remove = value_counts[value_counts < upper_cutoff].index

            to_remove = set(to_remove)

            # Create a boolean mask for the rows that do not contain rare molecules
            mask = ~df[col].isin(to_remove)

            # Create a new DataFrame with the selected rows
            df = df.loc[mask]

            post_len = len(df)

        return df

    def _get_dataframe(self) -> pd.DataFrame:
        # Merge all the pickled data into one big df
        df = self.merge_pickles()
        LOG.info(f"All data length: {len(df)}")

        # Remove reactions with too many of a certain component
        columns = [
            "reactant",
            "product",
            "yield",
            "solvent",
            "agent",
            "catalyst",
            "reagent",
        ]
        for col in columns:
            df = self.remove_reactions_with_too_many_of_component(df, col)
            if col != "yield":
                LOG.info(f"After removing reactions with too many {col}s: {len(df)}")

        # Ensure consistent yield
        if self.consistent_yield:
            # Keep rows with yield <= 100 or missing yield values
            mask = pd.Series(data=True, index=df.index)  # start with all rows selected
            for i in range(self.num_product):
                yield_col = "yield_" + str(i)
                yield_mask = (df[yield_col] >= 0) & (df[yield_col] <= 100) | pd.isna(
                    df[yield_col]
                )
                mask &= yield_mask

            df = df[mask]

            # sum of yields should be between 0 and 100
            yield_columns = df.filter(like="yield").columns

            # Compute the sum of the yield columns for each row
            df["total_yield"] = df[yield_columns].sum(axis=1)

            # Filter out reactions where the total_yield is less than or equal to 100, or is NaN or None
            mask = (
                (df["total_yield"] <= 100)
                | pd.isna(df["total_yield"])
                | pd.isnull(df["total_yield"])
            )
            df = df[mask]

            # Drop the 'total_yield' column from the DataFrame
            df = df.drop("total_yield", axis=1)
            LOG.info(f"After removing reactions with inconsistent yields: {len(df)}")

        # Remove reactions with rare molecules
        # Apply this to each column (this implies that if our cutoff is 100, and there's 60 instances of a molecule in one column,
        # and 60 instances of the same molecule in another column, we will still remove the reaction)

        # Get a list of columns with either solvent, reagent, catalyst, or agent in the name
        columns = []
        for col in list(df.columns):
            if col in ["reagent", "solvent", "catalyst", "agent"]:
                columns += [col]

        if (
            self.min_frequency_of_occurance_primary
            or self.min_frequency_of_occurance_secondary != 0
        ):
            df = self.remove_rare_molecules(df, columns)

        # cols = []
        # for col in list(df.columns):
        #     if 'reagent' in col or 'solvent' in col or 'catalyst' in col or 'agent' in col:
        #         cols += [col]
        # It may be faster to only loop over columns containing cat, solv, reag, or agent, however, if time isn't an issue we might as well loop over the whole df.

        for col in tqdm(df.columns):
            df = df[~df[col].isin(self.molecules_to_remove)]
            # Remove reactions that are represented by a name instead of a SMILES string
            # NB: There are 74k instances of solution, 59k instances of 'ice water', and 36k instances of 'ice'. I'm not sure what to do with these. I have decided to stay on the safe side and remove any reactions that includes one of these. However, other researchers are welcome to revisit this assumption - maybe we can recover a lot of insightful reactions by replacing 'ice' with 'O' (as in, the SMILES string for water).

        LOG.info(
            f"After removing reactions with nonsensical/unresolvable names: {len(df)}"
        )

        # This method is apparently very slow
        # # Replace any instances of an empty string with None
        # df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

        # # Replace nan with None
        # df.replace(np.nan, None, inplace=True)

        # Replace any instances of an empty string with None
        df = df.applymap(
            lambda x: None if (isinstance(x, str) and x.strip() == "") else x
        )

        # Replace np.nan with None
        df = df.applymap(lambda x: None if pd.isna(x) else x)

        # drop duplicates
        df = df.drop_duplicates()
        LOG.info(f"After removing duplicates: {len(df)}")

        df.reset_index(inplace=True)
        return df


@click.command()
@click.option("--clean_data_path", type=str, default="./cleaned_USPTO.parquet")
@click.option("--pickles_path", type=str)
@click.option("--molecules_to_remove_path", type=str)
@click.option("--consistent_yield", type=bool, default=True)
@click.option("--num_reactant", type=int, default=5)
@click.option("--num_product", type=int, default=5)
@click.option("--num_solv", type=int, default=2)
@click.option("--num_agent", type=int, default=3)
@click.option("--num_cat", type=int, default=0)
@click.option("--num_reag", type=int, default=0)
@click.option("--min_frequency_of_occurance_primary", type=int, default=15)
@click.option("--min_frequency_of_occurance_secondary", type=int, default=15)
@click.option("--include_other_category", type=bool, default=True)
@click.option(
    "--map_rate_to_other",
    type=int,
    default=3,
    help="save the reaction: label the rare molecule with 'other' rather than removing it",
)
@click.option("--disable_tqdm", type=bool, default=False)
def main(
    clean_data_path: pathlib.Path,
    pickles_path: pathlib.Path,
    molecules_to_remove_path: pathlib.Path,
    consistent_yield: bool,
    num_reactant: int,
    num_product: int,
    num_solv: int,
    num_agent: int,
    num_cat: int,
    num_reag: int,
    min_frequency_of_occurance_primary: int,
    min_frequency_of_occurance_secondary: int,
    include_other_category: bool,
    map_rate_to_other: int,
    disable_tqdm: bool,
):
    """
    After running USPTO_extraction.py, this script will merge and apply further cleaning to the data.

        Example:

    python USPTO_cleaning.py --clean_data_file_name=cleaned_USPTO --consistent_yield=True --num_reactant=5 --num_product=5 --num_solv=2 --num_agent=3 --num_cat=0 --num_reag=0 --min_frequency_of_occurance_primary=15 --min_frequency_of_occurance_secondary=15 --include_other_category=True --save_with_label_called_other=10


        Args:

    1) clean_data_file_name: (str) The filepath where the cleaned data will be saved
    2) consistent_yield: (bool) Remove reactions with inconsistent reported yields (e.g. if the sum is under 0% or above 100%. Reactions with nan yields are not removed)
    3) - 8) num_reactant, num_product, num_solv, num_agent, num_cat, num_reag: (int) The number of molecules of that type to keep. Keep in mind that if merge_conditions=True in USPTO_extraction, there will only be agents, but no catalysts/reagents, and if merge_conditions=False, there will only be catalysts and reagents, but no agents. Agents should be seen as a 'parent' category of reagents and catalysts; solvents should fall under this category as well, but since the space of solvents is more well defined (and we have a list of the most industrially relevant solvents which we can refer to), we can separate out the solvents. Therefore, if merge_conditions=True, num_catalyst and num_reagent should be set to 0, and if merge_conditions=False, num_agent should be set to 0. It is recommended to set merge_conditions=True, as we don't believe that the original labelling of catalysts and reagents that reliable; furthermore, what constitutes a catalyst and what constitutes a reagent is not always clear, adding further ambiguity to the labelling, so it's probably best to merge these.
    9) min_frequency_of_occurance_primary: (int) The minimum number of times a molecule must appear in the dataset to be kept. Infrequently occuring molecules will probably add more noise than signal to the dataset, so it is best to remove them. Primary: refers to the first index of columns of that type, ie solvent_0, agent_0, catalyst_0, reagent_0
    10) min_frequency_of_occurance_secondary: (int) See above. Secondary: Any other columns than the first.
    11) include_other_category (bool): Will save reactions with infrequent molecules (below min_frequency_of_occurance_primary/secondary but above save_with_label_called_other) by mapping these molecules to the string 'other'
    12) save_with_label_called_other (int): Frequency cutoff (see above).

        Functionality:

    1) Merge the pickle files from USPTO_extraction.py into a df

    2) Remove reactions with too many reactants, products, sovlents, agents, catalysts, and reagents (num_reactant, num_product, num_solv, num_agent, num_cat, num_reag)
    3) Remove reactions with inconsistent yields (consistent_yield)
    4) Removal or remapping to 'other' of rare molecules
    5) Remove reactions that have a molecule represented by an unresolvable name. This is often an english name or a number.
    6) Remove duplicate reactions
    7) Pickle the final df

        Output:

    1) A pickle file containing the cleaned data

        NB:
    1) There are lots of places where the code where I use masks to remove rows from a df. These operations could also be done in one line, however, using an operation such as .replace is very slow, and one-liners with dfs can lead to SettingWithCopyWarning. Therefore, I have opted to use masks, which are much faster, and don't give the warning.
    """
    start_time = datetime.datetime.now()

    clean_data_path = pathlib.Path(clean_data_path)
    pickles_path = pathlib.Path(pickles_path)
    molecules_to_remove_path = pathlib.Path(molecules_to_remove_path)

    molecules_to_remove = pd.read_parquet(molecules_to_remove_path)

    assert (
        num_agent == 0 or num_cat == 0 and num_reag == 0
    ), "Invalid input: If merge_conditions=True in USPTO_extraction, then num_cat and num_reag must be 0. If merge_conditions=False, then num_agent must be 0."
    assert (
        min_frequency_of_occurance_primary > map_rate_to_other
        and min_frequency_of_occurance_secondary > map_rate_to_other
    ), "min_frequency_of_occurance_primary and min_frequency_of_occurance_secondary must be greater than save_with_label_called_other. Anything between save_with_label_called_other and min_frequency_of_occurance_primary/secondary will be set to 'other' if include_other_category=True."

    instance = Cleaner(
        pickles_path=pickles_path,
        consistent_yield=consistent_yield,
        num_reactant=num_reactant,
        num_product=num_product,
        num_solv=num_solv,
        num_agent=num_agent,
        num_cat=num_cat,
        num_reag=num_reag,
        min_frequency_of_occurance_primary=min_frequency_of_occurance_primary,
        min_frequency_of_occurance_secondary=min_frequency_of_occurance_secondary,
        include_other_category=include_other_category,
        map_rate_to_other=map_rate_to_other,
        molecules_to_remove=molecules_to_remove,
        disable_tqdm=disable_tqdm,
    )
    instance.cleaned_reactions.to_parquet(clean_data_path)

    end_time = datetime.datetime.now()
    LOG.info("Duration: {}".format(end_time - start_time))
