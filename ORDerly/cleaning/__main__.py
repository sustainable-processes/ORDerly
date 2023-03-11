import logging
import click
import datetime
import pathlib

import pandas as pd

from ORDerly.cleaning.cleaner import Cleaner

LOG = logging.getLogger(__name__)


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


if __name__ == "__main__":
    main()
