import os
import shutil
import tempfile
import unittest

import yaml

from data.make_dataset import create_empty_hdf5_files, generate_polony_data

# PolonyDataset,,
# from data import make_dataset

# from data.make_dataset import CONFIG_PATH
# from data import make_dataset

CONFIG_PATH = "src/config/config.yaml"

with open(CONFIG_PATH, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


class TestMakeDataset(unittest.TestCase):
    def setUp(self):
        self.temp_folder = tempfile.mkdtemp(dir=os.path.join("D:", "Temp"))

    def test_create_empty_hdf5_files(self):
        train_h5, valid_h5 = create_empty_hdf5_files(
            dataset_name=self.temp_folder,
            train_size=90,
            valid_size=10,
            img_size=config["square_size"],
            in_channels=2,
        )
        return train_h5, valid_h5

    def test_generate_polony_data(self):
        generate_polony_data(data_root=self.temp_folder)

    def tearDown(self):
        shutil.rmtree(self.temp_folder)


if __name__ == "__main__":
    unittest.main()
