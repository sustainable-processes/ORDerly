import os
import sys
import pathlib


def _get_orderly_data_path() -> pathlib.Path:
    file_path = sys.modules["orderly.data"].__file__
    if file_path is None:
        raise ValueError("The path for orderly.data was not found")
    return pathlib.Path(os.path.dirname(file_path))


def get_path_of_test_ords() -> pathlib.Path:
    return _get_orderly_data_path() / "ord_test_data"


def get_path_of_test_extracted_ords(trust_labelling: bool = False) -> pathlib.Path:
    trust_labelling_str = (
        "_trust_labelling" if trust_labelling else "_dont_trust_labelling"
    )
    return _get_orderly_data_path() / f"extracted_ord_test_data{trust_labelling_str}"


def get_path_of_molecule_names(trust_labelling: bool = False) -> pathlib.Path:
    trust_labelling_str = (
        "_trust_labelling" if trust_labelling else "_dont_trust_labelling"
    )
    return (
        _get_orderly_data_path()
        / f"extracted_ord_test_data{trust_labelling_str}"
        / "molecule_names"
    )
