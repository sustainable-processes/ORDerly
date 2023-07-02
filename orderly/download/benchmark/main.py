import pathlib
import zipfile
import click
import logging
import requests

from click_loglevel import LogLevel
import pandas as pd

LOG = logging.getLogger(__name__)


@click.command()
@click.option(
    "--benchmark_zip_file",
    type=str,
    default="orderly_benchmark.zip",
    show_default=True,
    help="zip file name",
)
@click.option(
    "--benchmark_directory",
    type=str,
    default="orderly_benchmark/",
    show_default=True,
    help="location for the zip file to be saved",
)
@click.option(
    "--version",
    type=int,
    default=2,
    show_default=True,
    help="version for the download",
)
@click.option(
    "--log_file",
    type=str,
    default="download.log",
    show_default=True,
    help="path for the log file for download",
)
@click.option("--log-level", type=LogLevel(), default=logging.INFO)
def download_benchmark_click(
    benchmark_zip_file: str,
    benchmark_directory: str,
    version: int,
    log_file: str,
    log_level: int,
) -> None:
    download_benchmark(
        benchmark_zip_file=benchmark_zip_file,
        benchmark_directory=pathlib.Path(benchmark_directory),
        version=version,
        log_file=pathlib.Path(log_file),
        log_level=log_level,
    )


def download_benchmark(
    benchmark_zip_file: str = "orderly_benchmark.zip",
    benchmark_directory: pathlib.Path = pathlib.Path("orderly_benchmark/"),
    version: int = 2,
    log_file: pathlib.Path = pathlib.Path("download.log"),
    log_level: int = logging.INFO,
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        encoding="utf-8",
        format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=log_level,
    )

    figshare_url = (
        f"https://figshare.com/ndownloader/articles/23298467/versions/{version}"
    )
    LOG.info(f"Downloading benchmark from {figshare_url} to {benchmark_zip_file}")
    r = requests.get(figshare_url, allow_redirects=True)
    with open(benchmark_zip_file, "wb") as f:
        f.write(r.content)

    LOG.info("Unzipping benchmark")
    benchmark_directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(benchmark_zip_file, "r") as zip_ref:
        zip_ref.extractall(benchmark_directory)


if __name__ == "__main__":
    download_benchmark()
    train_df = pd.read_parquet("orderly_benchmark/orderly_benchmark_train.parquet")
    test_df = pd.read_parquet("orderly_benchmark/orderly_benchmark_test.parquet")
