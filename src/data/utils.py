import hashlib
import os
import zipfile
from glob import glob
from typing import Union

import cv2
import gdown
import numpy as np
from PIL import Image
from roifile import roiread
from scipy.ndimage import gaussian_filter

IMG_SIZE = (2450, 1438)
SQUARE_SIZE = (316, 316)
MODEL_SIZE = SQUARE_SIZE

JSON_PATH = os.path.abspath(
    os.path.join("data", "dataset_files", "path_dict.json")
)


def remove_img_without_roi(location):
    img_location = os.path.join(location, "slides")
    image_list = glob(os.path.join(img_location, "*.tif"))

    error_list = []

    for roi_path in image_list:
        try:
            roiread(roi_path)[1]
        except Exception as e:
            print(f'Warning: "{e}". File {roi_path} was deleted')
            os.remove(roi_path)
            error_list.append(roi_path)
            continue
    with open("errors.txt", "w") as file:
        for path in error_list:
            file.write(path + "\n")


def get_and_unzip(url: str, location: str):
    """Extract a ZIP archive from given URL.

    Args:
        url: url of a ZIP file
        location: target location to extract archive in
    """
    dataset = gdown.download(id=url, fuzzy=True, output="polony.zip")
    dataset = zipfile.ZipFile(dataset)
    dataset.extractall(location)
    dataset.close()
    os.remove(dataset.filename)
    remove_img_without_roi(location)


def read_tiff(path, new_size=None):
    """
    path - Path to the multipage-tiff file
    """
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        if new_size is not None:
            images.append(np.array(img.resize((new_size[1], new_size[0]))))
        else:
            images.append(np.array(img))
    return np.array(images, dtype=np.float32) / 255


def get_roi_coordinates(
    roi_path: str, channel: Union[int, None] = None, counter: bool = False
):
    """
    From ROI file with image get arrays with coordinates
    Args:
        roi_path (str): path to ROI image
        channel (Union[int, None], optional): for which channel of the image
        you need to return an array with coordinates.
            Defaults to None - for both. Can be 1 or 2

    Returns:
        np.ndarray: Array with coordinates
    """
    try:
        img_roi = roiread(roi_path)[1]
    except ValueError as e:
        print(f'ERROR "{e}" in file {roi_path}')
        os.remove(roi_path)
        raise (e)
    counter_positions = img_roi.counter_positions
    subpixel_coordinates = img_roi.subpixel_coordinates

    if not counter:
        if channel is None:
            coordinates_1 = subpixel_coordinates[counter_positions == 1]
            coordinates_2 = subpixel_coordinates[counter_positions == 2]
            return coordinates_1, coordinates_2
        elif channel == 1 or channel == 2:
            return subpixel_coordinates[counter_positions == channel]
        else:
            raise ValueError("Invalid value for variable. Must be 1 or 2")
    else:
        counters = img_roi.counters
        if channel is None:
            coordinates_1 = subpixel_coordinates[counter_positions == 1]
            coordinates_2 = subpixel_coordinates[counter_positions == 2]
            return (
                coordinates_1,
                coordinates_2,
                counters[counter_positions == 1],
                counters[counter_positions == 2],
            )
        elif channel == 1 or channel == 2:
            return (
                subpixel_coordinates[counter_positions == channel],
                counters[counter_positions == channel],
            )
        else:
            raise ValueError("Invalid value for variable. Must be 1 or 2")


def create_density_roi(coordinates, size=IMG_SIZE, new_size=None, channel=1):
    if new_size is not None:
        density = np.zeros(new_size, dtype=np.float32)

        # calculate scale coeffs
        scale_y = new_size[0] / size[0]
        scale_x = new_size[1] / size[1]

    else:
        density = np.zeros(size, dtype=np.float32)

    # make a one-channel label array with 100 in dots positions
    for x, y in coordinates:
        if x < size[1] and y < size[0]:
            if new_size is not None:
                # pow to scales coeffs
                x = int(x * scale_x)
                y = int(y * scale_y)
            if x < 0 or y < 0:
                print(x, y)
                return False
            density[int(y), int(x)] = 100

    # generate a density map by applying a Gaussian filter
    density = gaussian_filter(density, sigma=(1, 1), order=0)

    return density


def grid_to_squares(path):
    # Uploading an image
    imgs = read_tiff(path)
    img = imgs[0]

    img_roi = roiread(path)

    points, counters = get_roi_coordinates(
        roi_path=path,
        # TODO Think about how to make it so that it is automatically
        # determined on which channels there are points
        channel=1,
        counter=True,
    )

    lines_coordinates = (
        img_roi[0].multi_coordinates.reshape(-1, 6).astype(int)[:, 1:3]
    )

    # square size
    square_size = lines_coordinates[1, 0] - lines_coordinates[0, 0]

    # Dividing coordinates into horizontal and vertical lines
    horizontal_lines = lines_coordinates[lines_coordinates[:, 1] == 0][:, 0]
    vertical_lines = lines_coordinates[lines_coordinates[:, 0] == 0][:, 1]

    squares = []  # List for storing full squares

    # Splitting an image into squares
    for x in horizontal_lines[:-1]:
        for y in vertical_lines[:-1]:
            points_condition = (
                (points[:, 1] >= y)
                & (points[:, 1] < y + square_size)
                & (points[:, 0] >= x)
                & (points[:, 0] < x + square_size)
            )

            square_points = points[points_condition]

            square_id = counters[points_condition]
            id_value, id_counts = np.unique(square_id, return_counts=True)

            if len(square_points) <= 1 or (
                len(id_value) > 1 and max(id_counts) - min(id_counts) <= 2
            ):
                continue

            true_square_id = id_value[id_counts == max(id_counts)].item()

            # CHECK AND BRING BACK TO SQUARE POINTS if points are out of bounds
            square_points = bring_back_points(
                true_square_id, points, counters, x, y, square_size
            )

            square_dict = dict()
            square = img[y : y + square_size, x : x + square_size]
            square_2c = imgs[:, y : y + square_size, x : x + square_size]

            square_dict["square"] = square
            square_dict["square_2c"] = square_2c
            square_dict["points"] = square_points - np.array(
                [x, y]
            )  # points in relative coordinates for a given square
            square_dict["square_coordinate"] = np.array(
                [x, y]
            )  # coordinates of the upper-left corner of the square
            square_dict["n_points"] = len(square_points)

            square_dict["label"] = create_density_roi(
                square_dict["points"], size=MODEL_SIZE
            )
            if square_dict["label"] is False:
                print(
                    "x ",
                    x,
                    "y ",
                    y,
                    path,
                    "id ",
                    true_square_id,
                    "id_value ",
                    id_value,
                )
            squares.append(square_dict)

    return squares


def bring_back_points(square_id, points, counters, x, y, square_size):
    """
    This function makes INPLACE changes in the list points
    Args:
        square_id - id of square
        points - list with coordinates (full list)
        counters - list with points id (same id -> same square)
    Return:
        list of correct points of square with square_id
    """
    true_points = points[
        counters == square_id
    ]  # square points by id of square
    points_condition = (
        (true_points[:, 1] >= y)
        & (true_points[:, 1] < y + square_size)
        & (true_points[:, 0] >= x)
        & (true_points[:, 0] < x + square_size)
    )

    points_anti_condition = np.invert(
        points_condition
    )  # bool list with true on points which is not in square

    if (
        np.sum(points_anti_condition) == 0
    ):  # check if we don't have any points out of square
        return true_points  # return list of square_points
    else:
        mistake_points = true_points[points_anti_condition]
        for i, (p_x, p_y) in enumerate(mistake_points):
            if p_x < x:
                mistake_points[i][0] = x
            if p_x >= x + square_size:
                mistake_points[i][0] = x + square_size - 0.5
            if p_y < y:
                mistake_points[i][1] = y
            if p_y >= y + square_size:
                mistake_points[i][1] = y + square_size - 0.5
        true_points[points_anti_condition] = mistake_points
        return true_points


def save_squares(location):
    folder_to_save = os.path.join(location, "squares")
    # create output folder if it does not exist
    os.makedirs(folder_to_save, exist_ok=True)


def count_data_size(image_list):
    counter = 0
    for img_path in image_list:
        squares_list = grid_to_squares(img_path)
        counter += len(squares_list)

    return counter


def rgb_to_gray(my_data: np.ndarray):
    example = np.transpose(my_data, (1, 2, 0))
    example = cv2.cvtColor(example, cv2.COLOR_RGB2GRAY)
    example = np.expand_dims(example, axis=0)
    return example


def mean_std(data_train):
    mean = np.array([0.0, 0.0])
    std = np.array([0.0, 0.0])
    for i in range(len(data_train)):
        mean += data_train[i][0].mean((1, 2))
        std += data_train[i][0].std((1, 2))
    return mean / len(data_train), std / len(data_train)


def delete_duplicates(image_folder):
    # Creating a dictionary where the key is the hash of the image,
    # and the value is a list of files with this hash
    image_hash_dict = {}

    # Going through all the files in the folder
    for root, dirs, files in os.walk(image_folder):
        if len(files) == 0:
            continue
        for filename in files:
            file_path = os.path.join(root, filename)

            # Skip the non-images
            try:
                with Image.open(file_path) as img:
                    image_hash = hashlib.md5(img.tobytes()).hexdigest()
            except Exception as e:
                print(f"Error during file processing {file_path}: {e}")
                continue

            # If the hash of the image is already in the dictionary,
            # add the file to the corresponding list
            if image_hash in image_hash_dict:
                image_hash_dict[image_hash].append(file_path)
            else:
                # If there is no hash, create a new entry in the dictionary
                image_hash_dict[image_hash] = [file_path]

    # Deleting duplicate files
    for image_hash, file_list in image_hash_dict.items():
        if len(file_list) > 1:
            print(f"Removing duplicates for the hash {image_hash}:")
            for file_path in file_list[1:]:
                os.remove(file_path)
                print(f"  Deleted file: {file_path}")

    print("Deletion completed.")
