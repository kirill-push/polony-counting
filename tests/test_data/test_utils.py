import os
import shutil
import tempfile

import numpy as np
import pytest
import yaml

from data.make_dataset import generate_polony_data
from data.utils import (
    create_density_roi,
    get_and_unzip,
    get_roi_coordinates,
    grid_to_squares,
    read_tiff,
    remove_img_without_roi,
)

CONFIG_PATH = "src/config/config.yaml"

with open(CONFIG_PATH, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


def test_remove_img_without_roi(tmp_path):
    generate_polony_data(
        data_root=tmp_path,
        id_list=["11qu58SyRl1VCnRN4ujQvmJXU5k0UTJPS"],
        delete_data=False,
    )
    remove_img_without_roi(location=tmp_path, remove=False)


def test_get_and_unzip(tmp_path):
    get_and_unzip(url="11qu58SyRl1VCnRN4ujQvmJXU5k0UTJPS", location=tmp_path)


def test_read_tiff():
    path_to_roi_img = "resources/raw/test/test_img.tif"
    img_with_new_size = read_tiff(path=path_to_roi_img, new_size=(10, 10))
    assert img_with_new_size.shape[1:] == (10, 10)
    assert isinstance(img_with_new_size, np.ndarray)

    img_without_new_size = read_tiff(path=path_to_roi_img, new_size=None)
    assert isinstance(img_without_new_size, np.ndarray)


@pytest.mark.parametrize("channel", [None, 1, 2])
@pytest.mark.parametrize("counter", [True, False])
def test_get_roi_coordinates(channel, counter):
    path_to_roi_img = "resources/raw/test/test_img.tif"
    ans = get_roi_coordinates(
        roi_path=path_to_roi_img,
        channel=channel,
        counter=counter,
    )
    if counter:
        assert isinstance(ans, tuple)
        assert len(ans) == 2
    elif channel is None:
        assert isinstance(ans, tuple)
    else:
        assert isinstance(ans, np.ndarray)


@pytest.mark.parametrize("new_size", [(200, 200), (10, 10), None])
def test_create_density_roi(new_size):
    path_to_roi_img = "resources/raw/test/test_img.tif"
    coordinates = get_roi_coordinates(
        roi_path=path_to_roi_img,
        channel=1,
        counter=False,
    )
    density = create_density_roi(coordinates=coordinates, new_size=new_size)
    if new_size is not None:
        assert density.shape == new_size
    else:
        assert density.shape == tuple(config["img_size"])


def test_grid_to_squares():
    path_to_roi_img = "resources/raw/test/test_img.tif"
    list_with_dicts = grid_to_squares(path=path_to_roi_img)
    assert isinstance(list_with_dicts, list)
    assert isinstance(list_with_dicts[0], dict)
