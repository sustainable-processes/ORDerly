import logging
import os
import json
from typing import List, Dict, Tuple, Optional
import dataclasses
import datetime
import pathlib
import click

import tqdm
import tqdm.contrib.logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
        self.axis_font_size = 16
        self.heading_fontsize = 18

        plt.rcParams.update(
            {
                "font.size": 16,  # Default font size
                "xtick.labelsize": 16,  # X-axis tick font size
                "ytick.labelsize": 16,  # Y-axis tick font size
                "legend.fontsize": 16,  # Legend font size
                "axes.labelsize": 18,  # X and Y axis label font size
            }
        )

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

    import matplotlib.ticker as ticker

    import matplotlib.ticker as ticker

    @staticmethod
    def plot_num_rxn_component(
        df: pd.DataFrame,
        col_starts_with: str,
        plot_output_path: pathlib.Path,
        num_columns: int = 6,
    ) -> None:
        # clear the figure
        plt.clf()
        col_subset = ORDerlyPlotter._get_columns_to_plot(df, col_starts_with)
        if len(col_subset) == 0:
            LOG.info(f"No columns found starting with {col_starts_with}")
            return
        df_subset = df[col_subset]
        counts = ORDerlyPlotter._count_strings(df_subset)

        def transform_list(at_least_counts: List[int]) -> List[int]:
            """
            Transforms a list of counts representing the number of rows with at least N strings
            to a list representing the number of rows with exactly N strings.

            :param at_least_counts: List[int], where each element represents the number of rows
                                    with at least N strings for index N.
            :return: List[int], where each element represents the number of rows with exactly
                    N strings for index N.
            """
            histogram_counts = []
            for i in range(len(at_least_counts)):
                if i == len(at_least_counts) - 1:
                    # For the last element, it's already the exact count
                    histogram_counts.append(at_least_counts[i])
                else:
                    # Subtract the sum of the remaining elements from the current element
                    histogram_counts.append(at_least_counts[i] - at_least_counts[i + 1])
            return histogram_counts

        histogram_counts = transform_list(counts)

        # also want the number of reactions with 0 of that component
        histogram_counts = [df.shape[0] - counts[0]] + histogram_counts

        plotting_subset = histogram_counts[:num_columns]
        if len(plotting_subset) < num_columns:
            plotting_subset += [0] * (num_columns - len(plotting_subset))
        # create a bar plot of string counts for each column
        plt.bar(range(num_columns), plotting_subset, color="grey", edgecolor="black")

        # set the x-axis tick labels to the column names
        # plt.xticks(range(len(self.columns_to_plot)), self.columns_to_plot, rotation=90)

        # set the plot title and axis labels
        # plt.title(f"Components per reaction") # Usually the title is added on top of the figure on overleaf, after (a)
        plt.ylabel(f"Number of reactions (1000s)")
        plt.xlabel(f"Number of {col_starts_with}s")

        # Format y-axis labels with commas and divide by 1000
        plt.gca().yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: f"{x/1000:,.0f}")
        )

        # # Add a horizontal line at df.shape[0]
        # plt.axhline(y=df.shape[0], color="red", linestyle="--")

        # # Add a legend and move it up
        # plt.legend(
        #     ["Before filtering", f"After filtering".capitalize()], loc=(0.5, 0.72)
        # )

        figure_file_path = plot_output_path / f"{col_starts_with}_counts.png"

        # save the plot to file
        plt.savefig(figure_file_path, bbox_inches="tight", dpi=300)
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
        plt.bar(
            frequency, num_reactions, width=freq_step, edgecolor="black", color="grey"
        )

        # set the plot title and axis labels
        # plt.title(f"Removing rare molecules") Title should be added on overleaf
        plt.ylabel(f"Number of reactions (1000s)")
        plt.xlabel(f"Minimum frequency of occurrence")

        # # Add a horizontal line at df.shape[0]
        # plt.axhline(y=total_num_reactions, color="red", linestyle="--")

        plt.gca().yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: f"{x/1000:,.0f}")
        )

        # # Add a legend
        # plt.legend(
        #     ["Total reactions", f"Number of reactions".capitalize()], loc="right"
        # )

        # Adjust the legend position
        # plt.legend.set_bbox_to_anchor((1, 0.8))

        figure_file_path = (
            plot_output_path / f"min_freq_{freq_step}_{freq_threshold}.png"
        )

        # save the plot to file
        plt.savefig(figure_file_path, bbox_inches="tight", dpi=600)

        return

    ####################################################################################################
    def plot_molecule_popularity_histograms(self) -> None:
        for molecule in [
            "reactant",
            "product",
            "solvent",
            "catalyst",
            "reagent",
            "agent",
        ]:
            if molecule + "_000" in self.df.columns:
                ORDerlyPlotter.plot_molecule_popularity_histogram(
                    self.df, molecule, self.plot_output_path
                )

    @staticmethod
    def plot_molecule_popularity_histogram(
        df: pd.DataFrame,
        molecule_type: str,
        plot_output_path: pathlib.Path,
        num_molecules_to_plot: int = 100,
    ) -> None:
        """
        Plot a histogram showing how often the most popular molecules in the dataset appear.
        """
        plt.clf()
        if molecule_type.lower() == "catalyst":
            # Define the list of columns to check
            columns_to_count_from = ORDerlyPlotter._get_columns_beginning_with_str(
                columns=df.columns,
                target_strings=(molecule_type, "reagent"),
            )
        else:
            # Define the list of columns to check
            columns_to_count_from = ORDerlyPlotter._get_columns_beginning_with_str(
                columns=df.columns,
                target_strings=(molecule_type,),
            )
        # Get the value counts for each column and remove "NULL"
        value_counts = ORDerlyPlotter._get_value_counts(df, columns_to_count_from).drop(
            "NULL", errors="ignore"
        )
        sub_value_counts = value_counts[:num_molecules_to_plot]

        if molecule_type.lower() == "product":
            divider = 1
        else:
            divider = 1000

        # Plot the results in thousands
        plt.bar(
            range(1, len(sub_value_counts) + 1),
            sub_value_counts / divider,
            edgecolor="black",
            color="grey",
        )
        # set the plot title and axis labels

        plt.title(f"Most popular {molecule_type}s")
        plt.xlabel(f"Popularity rank")

        # Get top 10 molecules
        top_10 = sub_value_counts.head(10)

        # Conditional formatting for products
        molecule_entries = []
        for molecule, count in top_10.items():
            if (
                molecule
                == "c1ccc([P](c2ccccc2)(c2ccccc2)[Pd]([P](c2ccccc2)(c2ccccc2)c2ccccc2)([P](c2ccccc2)(c2ccccc2)c2ccccc2)[P](c2ccccc2)(c2ccccc2)c2ccccc2)cc1"
            ):
                molecule = "tetrakistriphenylphosphine palladium"
            if molecule_type.lower() == "product":
                molecule_entries.append(f"{molecule}: {count}")
            else:
                molecule_entries.append(f"{molecule}: {count / 1000:.1f}k")
        top_10_list = "\n".join(molecule_entries)

        # Add top 10 list inside the plot, aligned to the left
        plt.text(
            0.95,
            0.65,
            f"Top 10 {molecule_type}s:\n{top_10_list}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="center",
            ha="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        figure_file_path = plot_output_path / f"{molecule_type}_popularity.png"

        # save the plot to file
        plt.savefig(figure_file_path, bbox_inches="tight", dpi=600)

        return


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
    "--plot_molecule_popularity_histograms",
    type=bool,
    default=False,
    show_default=True,
    help="If true, plots a histogram showing the popularity of the most commonly occurring molecules in the dataset",
)
@click.option(
    "--log_file",
    type=str,
    default="default_path_plot.log",
    show_default=True,
    help="path for the log file for cleaning",
)
def main_click(
    clean_data_path: pathlib.Path,
    plot_output_path: pathlib.Path,
    plot_num_rxn_components_bool: bool,
    plot_frequency_of_occurrence_bool: bool,
    freq_threshold: int,
    freq_step: int,
    plot_molecule_popularity_histograms: bool,
    log_file: pathlib.Path = pathlib.Path("plots.log"),
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
        plot_molecule_popularity_histograms=plot_molecule_popularity_histograms,
        log_file=_log_file,
    )


def main(
    clean_data_path: pathlib.Path,
    plot_output_path: pathlib.Path,
    plot_num_rxn_components_bool: bool,
    plot_frequency_of_occurrence_bool: bool,
    freq_threshold: int,
    freq_step: int,
    plot_molecule_popularity_histograms: bool,
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
    if plot_molecule_popularity_histograms:
        instance.plot_molecule_popularity_histograms()

    LOG.info(f"completed plots, saving to {plot_output_path}")

    end_time = datetime.datetime.now()
    LOG.info("Plotting complete, duration: {}".format(end_time - start_time))
