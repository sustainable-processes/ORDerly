import os
import sys
import pathlib


def get_path_of_test_ords():
    return (
        pathlib.Path(os.path.dirname(sys.modules["orderly.data"].__file__))
        / "ord_test_data"
    )


def get_path_of_test_extracted_ords():
    return (
        pathlib.Path(os.path.dirname(sys.modules["orderly.data"].__file__))
        / "extracted_ord_test_data"
    )
