import pytest
import pathlib
from typing import List, Any


@pytest.mark.parametrize(
    "input_list,target_list",
    (
        pytest.param(
            [1234, 12312, 523423],
            ["1234", "12312", "523423"],
            id="standard ints list",
            marks=pytest.mark.xfail(
                reason="TypeError: expected a string but got type(i)=<class 'int'> for i=1234"
            ),
        ),
        pytest.param(
            ["abd", "asdsad", "sads"], ["abd", "asdsad", "sads"], id="standard str list"
        ),
        pytest.param(
            [123.23, 65, "sads"],
            ["123.23", "65", "sads"],
            id="standard mixed list",
            marks=pytest.mark.xfail(
                reason="TypeError: expected a string but got type(i)=<class 'float'> for i=123.23"
            ),
        ),
        pytest.param(
            [123.23, 65, "sads"],
            ["123.23", "65"],
            id="failing standard mixed list",
            marks=pytest.mark.xfail(reason="AssertionError: lists are different"),
        ),
    ),
)
def test_read_write_list_as_csv(
    tmp_path: pathlib.Path, input_list: List[Any], target_list: List[str]
) -> None:
    import orderly.data.util

    save_path = tmp_path / "tmp_csv.csv"

    orderly.data.util.save_list(x=input_list, path=save_path)
    loaded_list = orderly.data.util.load_list(path=save_path)
    for i in loaded_list:
        assert isinstance(i, str)
    assert loaded_list == target_list, "lists are different"
