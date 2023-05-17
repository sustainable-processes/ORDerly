import logging
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

from orderly.types import *

@dataclasses.dataclass(kw_only=True)
class ConditionPrediction:
    """
    Class for training a condition prediction model.

    """

    train_data_path: pathlib.Path
    test_data_path: pathlib.Path

    def __post_init__(self) -> None:
        self.train_df = pd.read_parquet(self.train_data_path)
        self.test_df = pd.read_parquet(self.test_data_path)

    

    
            
    






@click.command()
@click.option(
    "--log_file",
    type=str,
    default="default_path_plot.log",
    show_default=True,
    help="path for the log file for model",
)
@click.option("--log-level", type=LogLevel(), default=logging.INFO)
def main_click(
    train_data_path: pathlib.Path,
    test_data_path: pathlib.Path,
    log_file: pathlib.Path = pathlib.Path("model.log"),
    log_level: int = logging.INFO,
) -> None:
    """
    After extraction and cleaning of ORD data, this will train a condition prediction model.

    
    """
    _log_file = pathlib.Path(plot_output_path) / f"plot.log"
    if log_file != "default_path_plot.log":
        _log_file = pathlib.Path(log_file)

    main(
        
        plot_waterfall_bool=plot_waterfall_bool,
        log_file=_log_file,
        log_level=log_level,
    )


def main(
    train_data_path: pathlib.Path,
    test_data_path: pathlib.Path,
    model_save_path: pathlib.Path,
    log_file: pathlib.Path = pathlib.Path("plots.log"),
    log_level: int = logging.INFO,
) -> None:
    """
    After extraction and cleaning of ORD data, this will train a condition prediction model.
    
    
    Functionality:
    1) Load the the train and test data
    2) Get the fingerprint to use as input (Morgan fp from rdkit). Generating FP is slow, so we do it once and save it.
    3) Apply OHE to the target variables
    3) Train the model (use tqdm to show progress)
        3.1) Save graphs of training & validation loss and accuracy
        3.2) Save the model
    4) Evaluate the model on the test data
        4.1) Save the test loss and accuracy in a log file
        
    We can then use the model to predict the condition of a reaction.
    
    """

    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        encoding="utf-8",
        format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=log_level,
    )

    if not isinstance(train_data_path, pathlib.Path):
        e = ValueError(f"Expect pathlib.Path: got {type(train_data_path)}")
        LOG.error(e)
        raise e
    if not isinstance(test_data_path, pathlib.Path):
        e = ValueError(f"Expect pathlib.Path: got {type(test_data_path)}")
        LOG.error(e)
        raise e

    model_save_path.mkdir(parents=True, exist_ok=True)

    start_time = datetime.datetime.now()

    LOG.info(f"Beginning model training, saving to {model_save_path}")
    instance = ConditionPrediction(
        train_data_path = train_data_path,
        test_data_path = test_data_path,
    )


    LOG.info(f"Completed model training, saving to {model_save_path}")

    end_time = datetime.datetime.now()
    LOG.info("Training complete, duration: {}".format(end_time - start_time))
