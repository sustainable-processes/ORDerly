import os
import sys
import pathlib


def get_path_of_test_ords():
    return (
        pathlib.Path(os.path.dirname(sys.modules["orderly.data"].__file__))
        / "ord_test_data"
    )


def get_path_of_test_extracted_ords(trust_labelling: bool = False):
    trust_labelling_str = (
        "_trust_labelling" if trust_labelling else "_dont_trust_labelling"
    )
    file_path = sys.modules["orderly.data"].__file__
    assert file_path is not None
    return (
        pathlib.Path(os.path.dirname(file_path))
        / f"extracted_ord_test_data{trust_labelling_str}"
    )


def get_path_of_molecule_names(trust_labelling: bool = False):
    trust_labelling_str = (
        "_trust_labelling" if trust_labelling else "_dont_trust_labelling"
    )
    file_path = sys.modules["orderly.data"].__file__
    assert file_path is not None
    return (
        pathlib.Path(os.path.dirname(file_path))
        / f"extracted_ord_test_data{trust_labelling_str}"
        / "molecule_names"
    )
