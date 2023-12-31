import json
import os
import shutil
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import yaml

from polony import PolonyDataset, generate_polony_data

CONFIG_PATH = "src/polony/config/config.yaml"

with open(CONFIG_PATH, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


def mock_get_and_unzip(url: str, location: str) -> None:
    os.makedirs(os.path.join(location), exist_ok=True)
    # for i in range(10):
    shutil.copy("resources/raw/test/test_img.tif", os.path.join(location, "test_0.tif"))


@pytest.mark.parametrize("new_size", [None, [100, 100]])
@pytest.mark.parametrize("is_squares", [True, False])
@pytest.mark.parametrize("evaluation", [True, False])
@pytest.mark.parametrize("mode", ["density", "classifier"])
def test_generate_polony_data_mock_download(
    tmp_path, new_size, is_squares, evaluation, mode
) -> None:
    # Create the directory if it doesn't exist
    with patch(
        "polony.make_dataset.get_and_unzip", side_effect=mock_get_and_unzip
    ) as mock_zip:
        generate_polony_data(
            data_root=tmp_path,
            id_list=["11qu58SyRl1VCnRN4ujQvmJXU5k0UTJPS"],
            delete_data=True,
            is_path=False,
            download=True,
            is_squares=is_squares,
            evaluation=evaluation,
            new_size=new_size,
            mode=mode,
        )
        assert os.path.exists(os.path.join(tmp_path, "polony"))
        mock_zip.assert_called_once()


def test_generate_polony_data(tmp_path):
    generate_polony_data(
        data_root=tmp_path,
        id_list=["11qu58SyRl1VCnRN4ujQvmJXU5k0UTJPS"],
        is_path=False,
        mode="classifier",
    )


# Fixture for creating a mock hdf5
@pytest.fixture
def mock_hdf5(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    h5_file = d / "test.hdf5"
    with h5py.File(h5_file, "w") as f:
        f.create_dataset("images", data=np.random.rand(10, 3, 100, 100))
        f.create_dataset("labels", data=np.random.randint(0, 2, (10, 1, 100, 100)))
        f.create_dataset("n_points", data=np.random.randint(1, 100, (10, 1)))
        f.create_dataset("path", data=np.arange(10).reshape(-1, 1))
        f.create_dataset("class", data=np.ones(shape=(10, 1)))
    return str(h5_file)


# Fixture for creating a mock JSON file
@pytest.fixture
def mock_json(tmp_path):
    json_file = tmp_path / "paths.json"
    with open(json_file, "w") as f:
        json.dump({str(i): f"path_{i}" for i in range(10)}, f)
    return str(json_file)


# Test for initialization
@pytest.mark.parametrize("mode", ["density", "classifier"])
def test_initialization(mock_hdf5, mock_json, mode) -> None:
    dataset = PolonyDataset(mock_hdf5, 0.5, 0.5, True, mock_json, mode)
    assert dataset.horizontal_flip == 0.5
    assert dataset.vertical_flip == 0.5
    assert dataset.to_gray is True
    assert isinstance(dataset.path_dict, dict)


# Test for length method
def test_length(mock_hdf5, mock_json) -> None:
    dataset = PolonyDataset(mock_hdf5, 0.5, 0.5, True, mock_json)
    assert len(dataset) == 10  # Assuming the mock HDF5 file has 10 items


# Test for getitem method
@pytest.mark.parametrize("flip", [0.0, 1.0])
@pytest.mark.parametrize("mode", ["density", "classifier"])
def test_getitem(mock_hdf5, mock_json, flip, mode) -> None:
    dataset = PolonyDataset(mock_hdf5, flip, flip, False, mock_json, mode)
    if mode == "density":
        image, label, n_points, path = dataset[0]
        assert label.shape == (1, 100, 100)
        assert isinstance(n_points, np.ndarray)
    elif mode == "classifier":
        image, square_class, path = dataset[0]
        assert isinstance(square_class, np.ndarray)
    assert image.shape == (3, 100, 100)
    assert isinstance(path, str)
