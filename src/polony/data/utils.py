import hashlib
import os
import zipfile
from glob import glob
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import gdown
import numpy as np
import yaml
from numpy.typing import NDArray
from PIL import Image
from roifile import roiread
from scipy.ndimage import gaussian_filter

current_script_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_script_path)
CONFIG_PATH = os.path.join(root_path, "config", "config.yaml")

with open(CONFIG_PATH, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


def remove_img_without_roi(location: str, remove: bool = True) -> None:
    """Find all images without roi data, write txt file with paths to problems.

    Args:
        location (str): dir with slides. In this location must be dir slides
            and inside must be all slides.
        remove (bool, optional): if True - remove all files without roi data.
            Defaults to True.
    """
    img_location = os.path.join(location, "slides")
    image_list = glob(os.path.join(img_location, "*.tif"))

    error_list = []

    for roi_path in image_list:
        try:
            roiread(roi_path)[1]
        except Exception as e:
            print(f'Warning: "{e}". File {roi_path} was deleted')
            if remove:
                os.remove(roi_path)
            error_list.append(roi_path)
            continue
    if error_list:
        if isinstance(config["errors_path"], List):
            errors_path = "/".join(config["errors_path"])
        elif isinstance(config["errors_path"], str):
            errors_path = config["errors_path"]

        with open(errors_path, "w") as file:
            for path in error_list:
                file.write(path + "\n")


def get_and_unzip(url: str, location: str) -> None:
    """Extract a ZIP archive from given URL.

    Args:
        url (str): google drive url id of a ZIP file
        location (str): target location to extract archive in
    """
    dataset = gdown.download(id=url, fuzzy=True, output="polony.zip")
    dataset = zipfile.ZipFile(dataset)
    dataset.extractall(location)
    dataset.close()
    os.remove(dataset.filename)
    remove_img_without_roi(location)


def read_tiff(path: str, new_size: Optional[Tuple[int]] = None) -> np.ndarray:
    """Read tiff file from path, make resize and return numpy array wth image.

    Args:
        path (str): Path to the multipage-tiff file.
        new_size (Optional[Tuple], optional): If not None - resize img
            to new_size. Defaults to None.

    Returns:
        np.ndarray: Return numpy array with image
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
) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64]]]:
    """
    From ROI file with image get arrays with coordinates
    Args:
        roi_path (str): path to ROI image
        channel (Union[int, None], optional): for which channel of the image
        you need to return an array with coordinates.
            Defaults to None - for both. Can be 1 or 2
        counter (False): return or not also array with points areas id

    Returns:
        NDArray[np.float64] or Tuple[NDArray[np.float64]]]:
            Numpy array with points coordinates
            or if counter == True
                tuple with 2 arrays: (point coordinates, point area id)
            or if channel is None
                point coordinates - tuple with two arrays
                point area id - tuple with two arrays
    """
    try:
        img_roi = roiread(roi_path)[1]
    except ValueError as e:
        print(f'ERROR "{e}" in file {roi_path}')
        # os.remove(roi_path)
        raise e
    counter_positions = img_roi.counter_positions
    subpixel_coordinates = img_roi.subpixel_coordinates

    if not counter:
        if channel is None:
            coordinates_1 = subpixel_coordinates[counter_positions == 1]
            coordinates_2 = subpixel_coordinates[counter_positions == 2]
            return (coordinates_1, coordinates_2)
        elif channel == 1 or channel == 2:
            return subpixel_coordinates[counter_positions == channel]
        else:
            raise ValueError("Invalid value for variable. Must be 1 or 2")
    counters = img_roi.counters
    if channel is None:
        coordinates_1 = subpixel_coordinates[counter_positions == 1]
        coordinates_2 = subpixel_coordinates[counter_positions == 2]
        return (
            (
                coordinates_1,
                coordinates_2,
            ),
            (
                counters[counter_positions == 1],
                counters[counter_positions == 2],
            ),
        )
    if channel in (1, 2):  # skipcq: PYL-R1714
        return (
            subpixel_coordinates[counter_positions == channel],
            counters[counter_positions == channel],
        )
    else:
        raise ValueError("Invalid value for variable. Must be 1 or 2")


def create_density_roi(
    coordinates: np.ndarray,
    size: Tuple[int] = config["img_size"],
    new_size: Optional[Tuple[int]] = None,
) -> NDArray[np.float64]:
    """Create density map by points coordinates

    Args:
        coordinates (np.ndarray): Points coordinates
        size (Tuple[int], optional): Size of image.
            Defaults to config['img_size'].
        new_size (Optional[Tuple[int]], optional): Size for resize.
            Defaults to None.

    Returns:
        NDArray[np.float64]: Output density map
    """
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


def reshape_numpy(arr: np.ndarray, new_size: List[int]) -> np.ndarray:
    """Reshape numpy arrays (3d and 2d)

    Args:
        arr (np.ndarray): source vector
        new_size (List[int]): new size to reshape

    Returns:
        np.ndarray: numpy array after reshape to new_size
    """
    dim = len(arr.shape)
    if dim not in [2, 3]:
        raise ValueError("Shape mismatch")
    h, w = new_size[::-1]
    if dim == 3:
        new_arr = np.transpose(arr, (1, 2, 0))
        new_arr = cv2.resize(new_arr, (w, h))
        return np.transpose(new_arr, (2, 0, 1))
    return cv2.resize(arr, (w, h))


def grid_to_squares(
    path: str,
    new_size: Optional[List[int]] = None,
    mode: str = "density",
) -> List[Dict[str, Any]]:
    """Make dict with squares and other data for model training
        from original tiff file with image

    Args:
        path (str): path to tiff file with image
        new_size (List[int] | None): new_size for squares
        mode (str): for which model should be prepared squares 'density' or 'classifier'

    Returns:
        List[Dict[str, Any]]: List with dicts. Each dict for square image and
            training data for it
    """
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

    lines_coordinates = img_roi[0].multi_coordinates.reshape(-1, 6).astype(int)[:, 1:3]

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

            # skip squares with 1 or 0 points or with more than 2 types of points
            if len(square_points) == 1 or (
                len(id_value) > 1 and max(id_counts) - min(id_counts) <= 2
            ):
                continue
            elif len(square_points) == 0:
                square_class = 0
                # skip empty squares for 'density' mode
                true_square_id = -1
                if mode == "density":
                    continue
            else:
                square_class = 1
                true_square_id = id_value[id_counts == max(id_counts)].item()

            # CHECK AND BRING BACK TO SQUARE POINTS if points are out of bounds
            # ###TODO: (28.11) may be better idea to delete such squares
            square_points = bring_back_points(
                true_square_id, points, counters, x, y, square_size
            )

            square_dict = dict()
            square = img[y : y + square_size, x : x + square_size]
            square_2c = imgs[:, y : y + square_size, x : x + square_size]

            if new_size is None:
                square_dict["square"] = square
                square_dict["square_2c"] = square_2c
            else:
                square = np.transpose(square)
                square_dict["square"] = reshape_numpy(square, new_size)
                square_dict["square_2c"] = reshape_numpy(square_2c, new_size)
            # points in relative coordinates for a given square
            square_dict["points"] = square_points
            # coordinates of the upper-left corner of the square
            square_dict["square_coordinate"] = np.array([x, y])
            square_dict["n_points"] = len(square_points)

            square_dict["label"] = create_density_roi(
                square_dict["points"], size=[square_size] * 2, new_size=new_size
            )
            square_dict["class"] = square_class
            if square_dict["label"] is False and square_class == 1:
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


def bring_back_points(
    square_id: int,
    points: NDArray[np.float64],
    counters: NDArray[np.float64],
    x: int,
    y: int,
    square_size: int,
) -> NDArray[np.float64]:
    """This function makes INPLACE changes in the list points

    Args:
        square_id (int): id of square
        points (NDArray[np.float64]): list with coordinates (full list)
        counters (NDArray[np.float64]): list with points id (same id -> same square)
        x (int): x coordinate of square
        y (int): y coordinate of square
        square_size (int): size of square

    Returns:
        NDArray[np.float64]: list of correct points of square with square_id
    """
    if square_id == -1:
        return np.array([])
    true_points = points[counters == square_id]  # square points by id of square
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
        return true_points - np.array([x, y])  # return list of square_points
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
    return true_points - np.array([x, y])


def count_data_size(image_list, mode="density"):
    counter = 0
    for img_path in image_list:
        squares_list = grid_to_squares(img_path, mode=mode)
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
