import argparse
import json
import os
import shutil
from glob import glob
from random import random
from typing import Iterable, List, Optional

import h5py
import numpy as np
import yaml
from PIL import Image
from torch.utils.data import Dataset

from .utils import (
    count_data_size,
    create_density_roi,
    delete_duplicates,
    get_and_unzip,
    get_roi_coordinates,
    grid_to_squares,
    read_tiff,
    rgb_to_gray,
)

# folder to load config file
current_script_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_script_path)
CONFIG_PATH = os.path.join(root_path, "config", "config.yaml")

with open(CONFIG_PATH, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

if isinstance(config["json_path"], List):
    JSON_PATH = "/".join(config["json_path"])
elif isinstance(config["json_path"], str):
    JSON_PATH = config["json_path"]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "--train_size",
    "-s",
    type=int,
    default=config["generate_polony_data"]["train_size"],
    help="Percentage of the sample that is used for training",
)

parser.add_argument(
    "--download",
    "-d",
    type=bool,
    default=config["generate_polony_data"]["download"],
    help="Download data or don't",
)

parser.add_argument(
    "--path_to_data",
    "-p",
    type=str,
    default=config["path_to_data"],
    help="Path to dir with data or to dir for download data",
)

parser.add_argument(
    "--id_list_path",
    "-i",
    type=str,
    default=config["id_list_path"],
    help="Path to file with id list for gdown download",
)


class PolonyDataset(Dataset):
    """PyTorch dataset for HDF5 files generated with `get_data.py`."""

    def __init__(
        self,
        dataset_path: List[str] | str,
        horizontal_flip: float,
        vertical_flip: float,
        to_gray: bool,
        json_path: List[str] | str,
        mode: str = "density",
    ):
        """
        Initialize flips probabilities and pointers to a HDF5 file.

        Args:
            dataset_path: a path to a HDF5 file
            horizontal_flip: the probability of applying horizontal flip
            vertical_flip: the probability of applying vertical flip
            mode (str): for which model Dataset - 'density' or 'classifier'
        """
        super(PolonyDataset, self).__init__()
        if isinstance(dataset_path, List):
            self.dataset_path = "/".join(dataset_path)
        elif isinstance(dataset_path, str):
            self.dataset_path = dataset_path
        self.h5 = h5py.File(self.dataset_path, "r")
        self.images = self.h5["images"]
        self.labels = self.h5["labels"]
        self.n_points = self.h5["n_points"]
        self.path_id = self.h5["path"]
        self.square_class = self.h5["class"]
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.to_gray = to_gray
        if isinstance(json_path, List):
            self.json_path = "/".join(json_path)
        elif isinstance(json_path, str):
            self.json_path = json_path
        with open(self.json_path, "r") as file:
            self.path_dict = json.load(file)
        self.mode = mode

    def __len__(self):
        """Return no. of samples in HDF5 file."""
        return len(self.images)

    def pos_weight(self) -> float:
        """Returns a weight of positive examples for BCELoss

        Returns:
            float: a weight of positive examples for BCELoss
        """
        pos = 0.0
        neg = 0.0
        for i in range(len(self)):
            if self[i][1].item() == 0:
                neg += 1
            else:
                pos += 1
        self.postive_objects = pos
        self.negative_objects = neg
        return pos / neg

    def __getitem__(self, index: int):
        """Return next sample (randomly flipped)."""
        # if both flips probabilities are zero return an image and a label
        if not (self.horizontal_flip or self.vertical_flip):
            image = self.images[index]
            label = self.labels[index]
            n_points = self.n_points[index]
            path = self.path_dict[str(int(self.path_id[index][0]))]
            if self.to_gray:
                image = rgb_to_gray(image)
            if self.mode == "classifier":
                square_class = self.square_class[index]
                return image, square_class, path
            return image, label, n_points, path

        # axis = 1 (vertical flip), axis = 2 (horizontal flip)
        axis_to_flip = []

        if random() < self.vertical_flip:
            axis_to_flip.append(1)

        if random() < self.horizontal_flip:
            axis_to_flip.append(2)
        if self.mode == "classifier":
            return (
                np.flip(self.images[index], axis=axis_to_flip).copy(),
                self.square_class[index],
                self.path_dict[str(int(self.path_id[index][0]))],
            )
        return (
            np.flip(self.images[index], axis=axis_to_flip).copy(),
            np.flip(self.labels[index], axis=axis_to_flip).copy(),
            self.n_points[index],
            self.path_dict[str(int(self.path_id[index][0]))],
        )


def create_empty_hdf5_files(
    dataset_name: str,
    train_size: Optional[int],
    valid_size: int,
    img_size: List[int],
    in_channels: int,
    root_path: str,
):
    """
    Create empty training and validation HDF5 files (one file for training and
    one for validation) with placeholders for images and labels (density maps).

    Note:
    Datasets are saved in [dataset_name]/train.h5 and [dataset_name]/valid.h5.
    Existing files will be overwritten.

    Args:
        dataset_name: used to create a folder for train.h5 and valid.h5
        train_size: no. of training samples, if None -> evaluation mode
        valid_size: no. of validation samples
        img_size: (width, height) of a single image / density map
        in_channels: no. of channels of an input image
        root_path: path to root dir for h5 files

    Returns:
        A tuple of pointers to training and validation HDF5 files.
    """
    # create output folder if it does not exist
    dataset_path = os.path.join(root_path, dataset_name)
    os.makedirs(dataset_path, exist_ok=True)

    # if files exist - delete them
    if os.path.exists(os.path.join(dataset_path, "train.h5")):
        os.remove(os.path.join(dataset_path, "train.h5"))
    if os.path.exists(os.path.join(dataset_path, "valid.h5")):
        os.remove(os.path.join(dataset_path, "valid.h5"))

    # create HDF5 files: [dataset_path]/(train | valid).h5
    if train_size is not None:
        train_h5 = h5py.File(os.path.join(dataset_path, "train.h5"), "w")
    else:
        train_h5 = None
    valid_h5 = h5py.File(os.path.join(dataset_path, "valid.h5"), "w")

    # add two HDF5 datasets (images and labels) for each HDF5 file
    for h5, size in ((train_h5, train_size), (valid_h5, valid_size)):
        if h5 is not None:
            h5.create_dataset("images", (size, in_channels, *img_size))
            h5.create_dataset("path", (size, 1))
            h5.create_dataset("labels", (size, 1, *img_size))
            h5.create_dataset("n_points", (size, 1)),
            h5.create_dataset("class", (size, 1)),

    return train_h5, valid_h5


def generate_polony_data(
    download: bool = config["generate_polony_data"]["download"],
    train_size: int = config["generate_polony_data"]["train_size"],
    new_size: Optional[List[int]] = config["generate_polony_data"]["new_size"],
    data_root: str = config["generate_polony_data"]["data_root"],
    is_squares: bool = config["generate_polony_data"]["is_squares"],
    id_list: Optional[Iterable[str]] = config["generate_polony_data"]["id_list"],
    channels: int = config["generate_polony_data"]["channels"],
    evaluation: bool = config["generate_polony_data"]["evaluation"],
    delete_data: bool = config["generate_polony_data"]["delete_data"],
    is_path: bool = config["generate_polony_data"]["is_path"],
    mode: str = "density",
):
    """
    Generate HDF5 files for polony dataset.

    Args:
        id: zip id on gdrive
        train_size: % from 0 to 100 from data for train
        new_size: - if not None, then the new image size is specified
        download: bool - download data or no
        is_squares: bool - divide into squares or no
        mode: str - for which model ('density' or 'classifier') should be prepared data
    """
    if mode == "classifier":
        is_squares = True
        new_size = [256] * 2
    if not download:
        delete_data = False
    # download and extract dataset
    data_path = os.path.join(data_root, "polony")
    if download:
        for i, id_to_zip in enumerate(id_list):
            location = os.path.join(data_path, str(i))
            get_and_unzip(id_to_zip, location=location)
        delete_duplicates(data_path)

    if new_size is None:
        if is_squares:
            img_size = config["square_size"]
        else:
            img_size = config["img_size"]
    else:
        img_size = new_size

    image_list = []
    for i in range(len(os.listdir(data_path))):
        image_list += glob(os.path.join(data_path, str(i), "slides", "*.tif"))

    names_list = np.array([s.split("/")[-1] for s in image_list])
    sort_idx = names_list.argsort()
    image_list = np.array(image_list)[sort_idx]

    if is_squares:
        # count the number of squares in all images that contain dots
        n_data = count_data_size(image_list, mode=mode)
        if not evaluation:
            train_size = int((train_size / 100) * n_data)
            valid_size = n_data - train_size
            print(f"Data size {n_data}, training size {train_size}")
        else:
            train_size = None
            valid_size = n_data
            print(f"Data size {n_data}")
    else:
        n_data = len(image_list)
        if not evaluation:
            train_size = int((train_size / 100) * n_data)
            valid_size = n_data - train_size
        else:
            train_size = None
            valid_size = n_data

    # create training and validation HDF5 files
    train_h5, valid_h5 = create_empty_hdf5_files(
        dataset_name="polony",
        train_size=train_size,
        valid_size=valid_size,
        img_size=img_size,
        in_channels=channels,
        root_path=data_root,
    )

    # creating a dictionary of paths to collect information in the process of
    # launching fill_h5 function
    path_dict = dict()

    def fill_h5(
        h5,
        images,
        new_size=new_size,
        is_squares=is_squares,
        h5_val=None,
        train_size=train_size,
        valid_size=valid_size,
        channels=channels,
        mode=mode,
    ):
        """
        Save images and labels in given HDF5 file.

        Args:
            h5: HDF5 file
            images: the list of images paths
        """
        if h5 is None:
            h5 = h5_val
            h5_val = None
            train_size = valid_size
        train_j = 0
        val_j = 0
        for i, img_path in enumerate(images):
            # collecting a dictionary of paths {id: path_to_image}
            # (for future analysis of results)
            if i not in path_dict:
                path_dict[i] = img_path
            if is_squares:
                squares_list = grid_to_squares(img_path, new_size=new_size, mode=mode)
                for tt, square_dict in enumerate(squares_list):
                    if channels == 1:
                        image = square_dict["square"]
                    elif channels == 2:
                        image = square_dict["square_2c"]

                    label = square_dict["label"]
                    # get number of points for image
                    n_points = square_dict["n_points"]

                    # for classifier
                    square_class = square_dict["class"]

                    if train_j < train_size:
                        # save data to HDF5 file
                        h5["images"][train_j] = image
                        h5["path"][train_j] = i
                        h5["labels"][train_j, 0] = label
                        h5["n_points"][train_j] = n_points
                        # for classifier
                        h5["class"][train_j] = square_class
                        train_j += 1
                    elif h5_val is not None:
                        h5_val["images"][val_j] = image
                        h5_val["path"][val_j] = i
                        h5_val["labels"][val_j, 0] = label
                        h5_val["n_points"][val_j] = n_points
                        # for classifier
                        h5_val["class"][val_j] = square_class
                        val_j += 1
            else:
                # get an image as numpy array
                if new_size is None:
                    if channels == 1:
                        image = np.array(Image.open(img_path), dtype=np.float32) / 255
                    elif channels == 2:
                        image = read_tiff(img_path)
                else:
                    if channels == 1:
                        image = Image.open(img_path)
                        image = image.resize((new_size[1], new_size[0]))
                        image = np.array(image, dtype=np.float32) / 255
                    elif channels == 2:
                        image = read_tiff(img_path, new_size=new_size)
                # add dim=0 to shape
                if channels == 1:
                    image = np.expand_dims(image, 0)

                # load an RGB image
                coordinates = get_roi_coordinates(img_path, channel=1)
                label = create_density_roi(coordinates, new_size=new_size)

                # get number of points for image
                # TODO make getting the number of points
                n_points = len(coordinates)

                # save data to HDF5 file
                h5["images"][i] = image
                h5["labels"][i, 0] = label
                h5["n_points"][i] = n_points
        print(f"Train size {train_j}, validation size {val_j}")

    if is_squares:
        fill_h5(train_h5, image_list, h5_val=valid_h5)
    else:
        if train_h5 is not None:
            fill_h5(train_h5, image_list[:train_size])
        fill_h5(valid_h5, image_list[train_size:])

    if is_path:
        # writing a path_dict to a json file for further use
        with open(JSON_PATH, "w") as file:
            json.dump(path_dict, file)

    # close HDF5 files
    if train_h5 is not None:
        train_h5.close()
    valid_h5.close()

    if delete_data:
        try:
            for i in range(len(os.listdir(data_path))):
                shutil.rmtree(os.path.join(data_path, str(i)))
        except FileNotFoundError:
            print("File already was deleted")


def main(args):
    id_list = []
    if args.download and os.path.exists(args.id_list_path):
        with open(args.id_list_path, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
    elif args.download:
        id_list = config.id_list

    generate_polony_data(
        download=args.download,
        train_size=args.train_size,
        id_list=id_list,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
