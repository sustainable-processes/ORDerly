import logging
import os
import json
from typing import List, Dict, Tuple, Optional
import dataclasses
import datetime
import pathlib
import click

from click_loglevel import LogLevel

import tqdm
import tqdm.contrib.logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from orderly.types import *

LOG = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class ORDerlyPlotter:
    """
    Class for generating plots from the ORDerly dataset

    Output:

    1) png figures that are used in the ORDerly paper

    Args:

    """

    clean_data_path: pathlib.Path
    plot_output_path: pathlib.Path
    freq_threshold: int
    freq_step: int

    def __post_init__(self) -> None:
        self.df = pd.read_parquet(self.clean_data_path)

    ####################################################################################################

    def plot_num_rxn_components(self) -> None:
        for molecule in [
            "reactant",
            "product",
            "solvent",
            "catalyst",
            "reagent",
            "agent",
        ]:
            ORDerlyPlotter.plot_num_rxn_component(
                self.df, molecule, self.plot_output_path
            )

    @staticmethod
    def plot_num_rxn_component(
        df: pd.DataFrame,
        col_starts_with: str,
        plot_output_path: pathlib.Path,
        num_columns: int = 5,
    ) -> None:
        # clear the figure
        plt.clf()
        col_subset = ORDerlyPlotter._get_columns_to_plot(df, col_starts_with)
        if len(col_subset) == 0:
            LOG.info(f"No columns found starting with {col_starts_with}")
            return
        df_subset = df[col_subset]
        counts = ORDerlyPlotter._count_strings(df_subset)

        plotting_subset = counts[:num_columns]
        # create a bar plot of string counts for each column
        plt.bar(
            range(1, num_columns + 1), plotting_subset
        )  # Adjusted to start at index 1

        # set the x-axis tick labels to the column names
        # plt.xticks(range(len(self.columns_to_plot)), self.columns_to_plot, rotation=90)

        # set the plot title and axis labels
        plt.title(f"Components per reaction")
        plt.ylabel(f"Number of reactions")
        plt.xlabel(f"Number of {col_starts_with}s")

        # Add a horizontal line at df.shape[0]
        plt.axhline(y=df.shape[0], color="red", linestyle="--")

        # Add a legend
        plt.legend(["Total reactions", f"{col_starts_with} counts".capitalize()])

        figure_file_path = plot_output_path / f"{col_starts_with}_counts.png"

        # save the plot to file
        plt.savefig(figure_file_path, bbox_inches="tight", dpi=600)
        return

    @staticmethod
    def _get_columns_to_plot(df: pd.DataFrame, col_starts_with: str) -> List[str]:
        cols = [col for col in df.columns if col.startswith(col_starts_with)]
        return cols

    @staticmethod
    def _count_strings(df: pd.DataFrame) -> List[int]:
        string_counts = []
        for col in df.columns:
            count = df[col].apply(lambda x: isinstance(x, str)).sum()
            string_counts.append(count)
        return string_counts

    ####################################################################################################
    def plot_frequency_of_occurrence(self) -> None:
        ORDerlyPlotter.plot_freq(
            self.df, self.plot_output_path, self.freq_threshold, self.freq_step
        )
        return

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
    def plot_freq(
        df: pd.DataFrame,
        plot_output_path: pathlib.Path,
        freq_threshold: int = 100,
        freq_step: int = 10,
    ) -> None:
        # clear the figure
        plt.clf()

        # Define the list of columns to check
        columns_to_count_from = ORDerlyPlotter._get_columns_beginning_with_str(
            columns=df.columns,
            target_strings=("agent", "solvent", "reagent", "catalyst"),
        )

        # Get the value counts for each column
        value_counts = ORDerlyPlotter._get_value_counts(df, columns_to_count_from)
        ORDerlyPlotter.plot_value_counts(value_counts, plot_output_path)
        total_num_reactions = df.shape[0]
        num_reactions = []
        frequency = []

        for i in tqdm.tqdm(range(0, freq_threshold + 1, freq_step)):
            # Remove the rare molecules
            filtered_df = ORDerlyPlotter._remove_rare_molecules(
                df, columns_to_count_from, value_counts, i
            )
            num_reactions.append(filtered_df.shape[0])
            frequency.append(i)

        # Plot the results
        plt.bar(frequency, num_reactions, width=freq_step, edgecolor="black")

        # set the plot title and axis labels
        plt.title(f"Removing rare molecules")
        plt.ylabel(f"Number of reactions")
        plt.xlabel(f"Minimum frequency of occurrence")

        # Add a horizontal line at df.shape[0]
        plt.axhline(y=total_num_reactions, color="red", linestyle="--")

        # Add a legend
        plt.legend(["Total reactions", f"Number of reactions".capitalize()])

        figure_file_path = (
            plot_output_path / f"min_freq_{freq_step}_{freq_threshold}.png"
        )

        # save the plot to file
        plt.savefig(figure_file_path, bbox_inches="tight", dpi=600)

        return

    @staticmethod
    def plot_value_counts(
        value_counts: pd.Series,
        plot_output_path: pathlib.Path,
        num_molecules_to_plot: int = 100,
    ) -> None:
        # clear the figure
        plt.clf()
        sub_value_counts = value_counts[:num_molecules_to_plot]
        # Plot the results
        plt.bar(
            range(1, len(sub_value_counts) + 1), sub_value_counts, edgecolor="black"
        )
        # set the plot title and axis labels

        plt.title(f"Frequency of occurrence of molecules")
        plt.ylabel(f"Number of occurrences of molecules")
        plt.xlabel(f"Molecules")

        figure_file_path = plot_output_path / f"value_counts.png"

        # save the plot to file
        plt.savefig(figure_file_path, bbox_inches="tight", dpi=600)

        # clear the figure
        plt.clf()

        return

    ####################################################################################################

    def plot_waterfall(self) -> None:
        # TODO: Though I'm not sure that a waterfall plot is actually the best way to show how data is filtered out by ORDerly. Perhaps better simply with a table?
        pass


@click.command()
@click.option(
    "--clean_data_path",
    type=str,
    default="data/orderly/orderly_ord.parquet",
    show_default=True,
    help="The filepath where the cleaned data will be loaded from",
)
@click.option(
    "--plot_output_path",
    type=str,
    default="data/orderly/",
    show_default=True,
    help="The filepath where the plots will be saved",
)
@click.option(
    "--plot_num_rxn_components_bool",
    type=bool,
    default=True,
    show_default=True,
    help="If true, plots the number of reactions with a given number of reactants, products, solvents, agents, catalysts, and reagents",
)
@click.option(
    "--plot_frequency_of_occurrence_bool",
    type=bool,
    default=False,
    show_default=True,
    help="If true, plots the frequency of occurrence of molecules in the dataset",
)
@click.option(
    "--freq_threshold",
    type=int,
    default=100,
    show_default=True,
    help="Highest min_frequency_of_occurrence to plot",
)
@click.option(
    "--freq_step",
    type=int,
    default=10,
    show_default=True,
    help="Size of the step when plotting impact of min_frequency_of_occurrence (between 0 and freq_threshold)",
)
@click.option(
    "--plot_waterfall_bool",
    type=bool,
    default=False,
    show_default=True,
    help="If true, plots a waterfall chart showing how many reactions are removed at each step of the cleaning process",
)
@click.option(
    "--log_file",
    type=str,
    default="default_path_plot.log",
    show_default=True,
    help="path for the log file for cleaning",
)
@click.option("--log-level", type=LogLevel(), default=logging.INFO)
def main_click(
    clean_data_path: pathlib.Path,
    plot_output_path: pathlib.Path,
    plot_num_rxn_components_bool: bool,
    plot_frequency_of_occurrence_bool: bool,
    freq_threshold: int,
    freq_step: int,
    plot_waterfall_bool: bool,
    log_file: pathlib.Path = pathlib.Path("plots.log"),
    log_level: int = logging.INFO,
) -> None:
    """
    After extraction and cleaning, this can generate the plots used in the ORDerly paper.

    Functionality:

    1) If plot_num_rxn_components_bool: plots the number of reactions with a given number of reactants, products, solvents, agents, catalysts, and reagents
    2) If plot_frequency_of_occurrence_bool: plots the frequency of occurrence of molecules in the dataset
    """
    _log_file = pathlib.Path(plot_output_path) / f"plot.log"
    if log_file != "default_path_plot.log":
        _log_file = pathlib.Path(log_file)

    main(
        clean_data_path=pathlib.Path(clean_data_path),
        plot_output_path=pathlib.Path(plot_output_path),
        plot_num_rxn_components_bool=plot_num_rxn_components_bool,
        plot_frequency_of_occurrence_bool=plot_frequency_of_occurrence_bool,
        freq_threshold=freq_threshold,
        freq_step=freq_step,
        plot_waterfall_bool=plot_waterfall_bool,
        log_file=_log_file,
        log_level=log_level,
    )


def main(
    clean_data_path: pathlib.Path,
    plot_output_path: pathlib.Path,
    plot_num_rxn_components_bool: bool,
    plot_frequency_of_occurrence_bool: bool,
    freq_threshold: int,
    freq_step: int,
    plot_waterfall_bool: bool,
    log_file: pathlib.Path = pathlib.Path("plots.log"),
    log_level: int = logging.INFO,
) -> None:
    """
    After extraction and cleaning, this can generate the plots used in the ORDerly paper.

    Functionality:

    1) If plot_num_rxn_components_bool: plots the number of reactions with a given number of reactants, products, solvents, agents, catalysts, and reagents
    2) If plot_frequency_of_occurrence_bool: plots the frequency of occurrence of molecules in the dataset
    """

    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        encoding="utf-8",
        format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=log_level,
    )

    if not isinstance(clean_data_path, pathlib.Path):
        e = ValueError(f"Expect pathlib.Path: got {type(clean_data_path)}")
        LOG.error(e)
        raise e
    if not isinstance(plot_output_path, pathlib.Path):
        e = ValueError(f"Expect pathlib.Path: got {type(plot_output_path)}")
        LOG.error(e)
        raise e

    plot_output_path.mkdir(parents=True, exist_ok=True)

    start_time = datetime.datetime.now()

    LOG.info(f"Beginning generation of plots for file: {clean_data_path}")
    instance = ORDerlyPlotter(
        clean_data_path=clean_data_path,
        plot_output_path=plot_output_path,
        freq_threshold=freq_threshold,
        freq_step=freq_step,
    )
    if plot_num_rxn_components_bool:
        instance.plot_num_rxn_components()
    if plot_frequency_of_occurrence_bool:
        instance.plot_frequency_of_occurrence()
    if plot_waterfall_bool:
        instance.plot_waterfall()

    LOG.info(f"completed plots, saving to {plot_output_path}")

    end_time = datetime.datetime.now()
    LOG.info("Cleaning complete, duration: {}".format(end_time - start_time))
