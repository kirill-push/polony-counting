"""
This script predicts the number of points in squares on images using
a pre-trained model.

The script takes two arguments:
  1. --path or -p: Path to the folder or file containing the images
    to be processed.
  2. --path_to_model or -m: Path to the saved model state dictionary used
    for predictions.

The prediction results are saved in a file named 'prediction' in a temporary
folder.

Usage example:
python predict_model.py --path /path/to/images --path_to_model /path/to/model

"""

import argparse
import os
import tempfile
from typing import Dict, List

import torch
from torchvision import transforms

from ..data.utils import read_tiff
from .models import UNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FIRST_HORIZONTAL = 158
FIRST_VERTICAL = 158
SQUARE_SIZE = [316, 316]
# [vertical lines, horizontal lines] = [8, 5] by ImageJ
NUMBER_OF_LINES = [8, 5]

MEAN, STD = ([12.69365111, 2.47628206], [13.35308926, 2.45260453])
normalize = transforms.Normalize(MEAN, STD)

# Configuring Argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "--path",
    "-p",
    type=str,
    default=".",
    help="Path to folder or file with images",
)
parser.add_argument(
    "--path_to_model",
    "-m",
    type=str,
    default="../checkpoints/polony_49_1.7496.pth",
    help="Path to saved state dict",
)


def predict(
    path: str,
    path_to_model: str = "..checkpoints/classifier_57_0.8896.pth",
    model: torch.nn.Module = UNet(res=False),
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
    channels: int = 2,
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
            predictions.append(
                predict_one_image(image_path, path_to_model, model, device, channels)
            )
    else:
        predictions.append(
            predict_one_image(path, path_to_model, model, device, channels)
        )

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
    square_size = SQUARE_SIZE[0]

    # vertical (8) and horizontal (5) lines
    v, h = NUMBER_OF_LINES
    horizontal_lines = [first_horizontal + square_size * i for i in range(h)]
    vertical_lines = [first_vertical + square_size * i for i in range(v)]

    # Devide image to squares and predict n points
    square_id = -1
    for x in horizontal_lines[:-1]:
        for y in vertical_lines[:-1]:
            result_dict = dict()
            square_id += 1
            if channels == 1:
                square = img[y : y + square_size, x : x + square_size]
            elif channels == 2:
                square = imgs[:, y : y + square_size, x : x + square_size]
            square = torch.from_numpy(square).float().to(device)
            square = normalize(square)
            density = network(square.unsqueeze(0))
            result = torch.sum(density).item() // 100

            result_dict["result"] = int(result)
            result_dict["density"] = density
            squares_dict[square_id] = result_dict

    return squares_dict


def main(args):
    temp_folder = tempfile.mkdtemp()
    print("Creating temporary folder: {}".format(temp_folder))

    predictions = predict(
        path=args.path,
        path_to_model=args.path_to_model,
    )

    with open(os.path.join(temp_folder, "prediction"), "w") as file:
        for pred in predictions:
            for k, v in pred.items():
                file.write(f"Square {k}, number of points {v}\n")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
