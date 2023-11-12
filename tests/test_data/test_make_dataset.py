import os

import yaml

from data.make_dataset import create_empty_hdf5_files, generate_polony_data

CONFIG_PATH = "src/config/config.yaml"

with open(CONFIG_PATH, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


def test_create_empty_hdf5_files(tmp_path):
    train_h5, valid_h5 = create_empty_hdf5_files(
        dataset_name="polony",
        train_size=90,
        valid_size=10,
        img_size=config["square_size"],
        in_channels=2,
        root_path=tmp_path,
    )
    assert os.path.exists(train_h5.filename)
    assert os.path.exists(valid_h5.filename)


def test_generate_polony_data(tmp_path):
    generate_polony_data(
        data_root=tmp_path,
        id_list=["11qu58SyRl1VCnRN4ujQvmJXU5k0UTJPS"],
        delete_data=False,
    )
    assert os.path.exists(os.path.join(tmp_path, "polony"))
