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

    def __post_init__(self) -> None:
        self.df = pd.read_parquet(self.clean_data_path)

    def plot_num_rxn_components(self):
        for molecule in [
            "reactant",
            "product",
            "solvent",
            "catalyst",
            "reagent",
            "agent",
        ]:
            self.plot_num_rxn_component(self.df,molecule, self.plot_output_path)

    @staticmethod
    def plot_num_rxn_component(self,df, col_starts_with, plot_output_path, num_columns=10):
        # clear the figure
        plt.clf()
        col_subset = ORDerlyPlotter._get_columns_to_plot(df)
        df_subset = df[col_subset]
        counts = ORDerlyPlotter._count_strings(df_subset)

        plotting_subset = counts[:num_columns]
        # create a bar plot of string counts for each column
        plt.bar(range(len(plotting_subset)), plotting_subset)

        # set the x-axis tick labels to the column names
        # plt.xticks(range(len(self.columns_to_plot)), self.columns_to_plot, rotation=90)

        # set the plot title and axis labels
        plt.title(
            f'Number of reactions with at least this many {col_starts_with}'
        )
        plt.ylabel(f"Number of reactions")
        plt.xlabel(f"Number of {col_starts_with}")

        figure_file_path = plot_output_path / f"{col_starts_with}_counts.png"

        # save the plot to file
        plt.savefig(figure_file_path, bbox_inches="tight")

    @staticmethod
    def _get_columns_to_plot(self, df, col_starts_with):
        cols = [col for col in df.columns if col.startswith(col_starts_with)]
        return cols

    @staticmethod
    def _count_strings(self, df):
        string_counts = []
        for col in tqdm(df.columns):
            count = df[col].apply(lambda x: isinstance(x, str)).sum()
            string_counts.append(count)
        return string_counts
    
    def plot_frequency_of_occurrence(self):
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
    overwrite: bool,
    log_file: pathlib.Path = pathlib.Path("plots.log"),
    log_level: int = logging.INFO,
) -> None:
    """
    After extraction and cleaning, this can generate the plots used in the ORDerly paper.

    Functionality:

    1) If plot_num_rxn_components_bool: plots the number of reactions with a given number of reactants, products, solvents, agents, catalysts, and reagents
    2) If plot_frequency_of_occurrence_bool: plots the frequency of occurrence of molecules in the dataset
    """
    _log_file = pathlib.Path(plot_output_path).parent / f"plot.log"
    if log_file != "default_path_plot.log":
        _log_file = pathlib.Path(log_file)

    main(
        clean_data_path=clean_data_path,
        plot_output_path=plot_output_path,
        plot_num_rxn_components_bool=plot_num_rxn_components_bool,
        plot_frequency_of_occurrence_bool=plot_frequency_of_occurrence_bool,
        log_file=_log_file,
        log_level=log_level,
    )


def main(
    clean_data_path: pathlib.Path,
    plot_output_path: pathlib.Path,
    plot_num_rxn_components_bool: bool,
    plot_frequency_of_occurrence_bool: bool,
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

    plot_output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = datetime.datetime.now()

    LOG.info(f"Beginning generation of plots for file: {clean_data_path}")
    instance = ORDerlyPlotter(
        clean_data_path=clean_data_path,
        plot_output_path=plot_output_path,
    )
    if plot_num_rxn_components_bool:
        instance.plot_num_rxn_components()
    if plot_frequency_of_occurrence_bool:
        instance.plot_frequency_of_occurrence()

    LOG.info(f"completed plots, saving to {plot_output_path}")

    end_time = datetime.datetime.now()
    LOG.info("Cleaning complete, duration: {}".format(end_time - start_time))