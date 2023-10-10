import os
import shutil
import tempfile
import unittest
from typing import Dict, List, Tuple

import numpy as np
import yaml

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


class TestTest(unittest.TestCase):
    def setUp(self):
        self.temp_folder = tempfile.mkdtemp(dir=os.path.join("D:", "Temp"))
        self.path_to_roi_img = "data/raw/slides/test_img.tif"

    def test_remove_img_without_roi(self):
        remove_img_without_roi(location="data/raw", remove=False)

    def test_get_and_unzip(self):
        get_and_unzip(
            url="1DhItO1ZGw1rXvYABff-bEumbAMRWifn0", location=self.temp_folder
        )

    def test_read_tiff(self):
        img_with_new_size = read_tiff(
            path=self.path_to_roi_img,
            new_size=(10, 10),
        )
        self.assertEqual(img_with_new_size.shape[1:], (10, 10))
        self.assertIsInstance(img_with_new_size, np.ndarray)

        img_without_new_size = read_tiff(
            path=self.path_to_roi_img,
            new_size=None,
        )
        self.assertIsInstance(img_without_new_size, np.ndarray)

    def test_get_roi_coordinates(self):
        for channel in [None, 1, 2]:
            for counter in [True, False]:
                ans = get_roi_coordinates(
                    roi_path=self.path_to_roi_img,
                    channel=channel,
                    counter=counter,
                )
                if counter:
                    self.assertIsInstance(ans, Tuple)
                    self.assertEqual(len(ans), 2)
                elif channel is None:
                    self.assertIsInstance(ans, Tuple)
                else:
                    self.assertIsInstance(ans, np.ndarray)

    def test_create_density_roi(self):
        coordinates = get_roi_coordinates(
            roi_path=self.path_to_roi_img,
            channel=1,
            counter=False,
        )
        for new_size in [(200, 200), (10, 10), None]:
            density = create_density_roi(
                coordinates=coordinates, new_size=new_size
            )
            if new_size is not None:
                self.assertEqual(density.shape, new_size)
            else:
                self.assertEqual(density.shape, tuple(config["img_size"]))

    def test_grid_to_squares(self):
        list_with_dicts = grid_to_squares(path=self.path_to_roi_img)
        self.assertIsInstance(list_with_dicts, List)
        self.assertIsInstance(list_with_dicts[0], Dict)

    def test_bring_back_points(self):
        ...

    def test_count_data_size(self):
        ...

    def test_rgb_to_gray(self):
        ...

    def test_mean_std(self):
        ...

    def test_delete_duplicates(self):
        ...

    def tearDown(self):
        shutil.rmtree(self.temp_folder)
