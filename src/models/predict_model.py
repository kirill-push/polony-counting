import os
from typing import Dict, List, Tuple

import torch

from data.utils import read_tiff
from models.models import UNet
from models.utils import grid_to_squares

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FIRST_HORIZONTAL = ...
FIRST_VERTICAL = ...
SQUARE_SIZE = 316
# [horizontal lines, vertical lines] = [8, 5] by ImageJ
NUMBER_OF_LINES = [8, 5]


def evaluate(
    path_to_example: str,
    path_to_model: str,
) -> Tuple[float, int]:
    """
    args:
        path_to_example - path to one image
        path_to_model - path to saved state dict of model
    return:
        mean absolute error on squares with points,
        total number of points in thpoetry run mypy image
    """

    network = torch.nn.DataParallel(UNet(res=False))
    network.load_state_dict(torch.load(path_to_model, map_location=device))
    network = network.eval()
    example = grid_to_squares(path_to_example)
    error = 0
    abs_result = 0
    for i in range(len(example)):
        img_1 = example[i]["square_2c"]
        img_1 = torch.from_numpy(img_1).float()
        density = network(img_1.unsqueeze(0))
        n_points = example[i]["n_points"]
        result = torch.sum(density).item() // 100
        error += abs(int(n_points) - int(result))
        abs_result += int(result)
    return error / len(example), abs_result


def predict(
    path: str,
    path_to_model: str,
    model: torch.nn.Module = UNet(res=False),
) -> List[Dict[int, int]]:
    """
    args:
        path: str - path to folder or to file
        path_to_model - path to saved state dict
        model - network which state dict was saved
    return:
        prediction - list with dictionaries;
                     one dict for one image;
                     dict = {square_id: number of points}
    """
    is_dir = os.path.isdir(path)

    predictions = []

    if is_dir:
        images = os.listdir(path)
        print("Files and folders in the directory:")
        for image_path in images:
            predictions.append(predict_one_image(image_path, path_to_model))
    else:
        predictions.append(predict_one_image(path, path_to_model))

    return predictions


def predict_one_image(
    path: str,
    path_to_model: str,
    model: torch.nn.Module = UNet(res=False),
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
    channels: int = 2,
) -> Dict[int, int]:
    """
    args:
        path - path to image
        path_to_model - path to saved state dict of model
        model - network which state dict was saved

        channels - 1 or 2 channels of image we need (depends on model)

    return:
        dictionary: {square_id: number_of_points}
    """
    network = torch.nn.DataParallel(model).to(device)
    network.load_state_dict(torch.load(path_to_model, map_location=device))
    network = network.eval()

    if channels not in [1, 2]:
        raise ValueError("Not correct value of channels, must be 1 or 2")

    imgs = read_tiff(path)
    img = imgs[0]

    squares_dict = dict()
    # TODO find mean value of first lines
    first_horizontal = FIRST_HORIZONTAL
    first_vertical = FIRST_VERTICAL

    # square size TODO find constant value of square_size
    square_size = SQUARE_SIZE

    # horizontal (8) and vertical (5) lines
    h, v = NUMBER_OF_LINES
    horizontal_lines = [first_horizontal + square_size * i for i in range(h)]
    vertical_lines = [first_vertical + square_size * i for i in range(v)]

    # Devide image to squares and predict n points
    square_id = -1
    for x in horizontal_lines[:-1]:
        for y in vertical_lines[:-1]:
            square_id += 1
            if channels == 1:
                square = img[y : y + square_size, x : x + square_size]
            elif channels == 2:
                square = imgs[:, y : y + square_size, x : x + square_size]
            square = torch.from_numpy(square).float().to(device)
            density = network(square.unsqueeze(0))
            result = torch.sum(density).item() // 100

            squares_dict[square_id] = int(result)

    return squares_dict
