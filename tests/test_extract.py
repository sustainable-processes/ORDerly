import pytest


def test_hello_world():
    assert True


def test_extract(tmp_path):

    pickled_data_folder = tmp_path / "pkl_data"
    pickled_data_folder.mkdir()
    molecule_names_folder = tmp_path / "molecule_names"
    molecule_names_folder.mkdir()

    import orderly.extract.main
    import orderly.data

    orderly.data.get_path_of_test_ords

    orderly.extract.main.main(
        data_path=str(orderly.data.get_path_of_test_ords()),
        ord_file_ending=".pb.gz",
        merge_conditions=True,
        output_path=tmp_path,
        pickled_data_folder=pickled_data_folder,
        molecule_names_folder=molecule_names_folder,
        merged_molecules_file="all_molecule_names.pkl",
        use_multiprocessing=False,
        name_contains_substring="uspto",
        overwrite=True,
    )