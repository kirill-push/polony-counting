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
from .models import Classifier, UNet
from .utils import logit_to_class

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FIRST_HORIZONTAL = 158
FIRST_VERTICAL = 158
SQUARE_SIZE = [316, 316]
# [vertical lines, horizontal lines] = [8, 5] by ImageJ
NUMBER_OF_LINES = [8, 5]

MEAN, STD = ([12.69365111, 2.47628206], [13.35308926, 2.45260453])
normalize = transforms.Normalize(MEAN, STD)

# paths to checkpoints
current_file_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_path)

density_checkpoint = os.path.join(project_root, "checkpoints", "unet_49_1.7496.pth")
classifier_checkpoint = os.path.join(
    project_root, "checkpoints", "classifier_57_0.8896.pth"
)

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
    default="../checkpoints/unet_49_1.7496.pth",
    help="Path to saved state dict",
)


def predict(
    path: str,
    density: torch.nn.Module = UNet(res=False),
    path_to_density: str = density_checkpoint,
    classifier: torch.nn.Module = Classifier(),
    path_to_classifier: str = classifier_checkpoint,
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
    channels: int = 2,
    classifier_threshold: float = 0.5,
) -> List[Dict[int, int]]:
    """Make prediction for all images from path folder or for one image from path.

    Args:
        path (str): path to folder or to file with images
        density (torch.nn.Module, optional): density network which state dict was saved.
            Defaults to UNet(res=False).
        path_to_density (str, optional): path to saved state dict for density model.
            Defaults to "../checkpoints/unet_49_1.7496.pth".
        classifier (torch.nn.Module, optional): classifier network to predict class.
            Defaults to Classifier().
        path_to_classifier (str, optional): path to saved weights of classifier model.
            Defaults to "../checkpoints/classifier_57_0.8896.pth".
        device (torch.device, optional): device for models.
            Defaults to torch.device("cuda:0" if torch.cuda.is_available() else "cpu").
        channels (int, optional): images channels.
            Defaults to 2.
        classifier_threshold (float, optional): threshold for classifier.
            Defaults to 0.5

    Raises:
        ValueError: Not correct value of input channels, must be 1 or 2.

    Returns:
        List[Dict[int, int]]: prediction - list with dictionaries;
            one dict for one image;
            dict = {square_id: number of points}
    """
    if channels not in [1, 2]:
        raise ValueError("Not correct value of channels, must be 1 or 2")
    is_dir = os.path.isdir(path)
    predictions = []
    density = torch.nn.DataParallel(density).to(device)
    density.load_state_dict(torch.load(path_to_density, map_location=device))
    density = density.eval()

    classifier = torch.nn.DataParallel(classifier).to(device)
    classifier.load_state_dict(torch.load(path_to_classifier, map_location=device))
    classifier = classifier.eval()

    if is_dir:
        images = os.listdir(path)
        print("Files and folders in the directory:")
        for image_path in images:
            predictions.append(
                predict_one_image(
                    image_path,
                    density,
                    classifier,
                    channels,
                    device,
                    classifier_threshold,
                )
            )
    else:
        predictions.append(
            predict_one_image(
                path, density, classifier, channels, device, classifier_threshold
            )
        )

    return predictions


def predict_one_image(
    path: str,
    network: torch.nn.Module,
    classifier: torch.nn.Module,
    channels: int,
    device: torch.device,
    classifier_threshold: float,
) -> Dict[int, int]:
    """Make prediction for one image from path.

    Args:
        path (str): path to one image
        network (torch.nn.Module, optional): density network.
        classifier (torch.nn.Module, optional): classifier network.
        channels (int, optional): image channels.
        device (torch.device, optional): device for models.
        classifier_threshold (float, optional): threshold for classifier.
    Returns:
        Dict[int, int]: dictionary with keys and values {square_id: number_of_points}
    """

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
            logit = classifier(square.unsqueeze(0))
            square_class = logit_to_class(logit, classifier_threshold).item()
            # if square_class == 0:
            #     continue
            density = network(square.unsqueeze(0))
            result = torch.sum(density).item() // 100

            result_dict["result"] = int(result)
            result_dict["class"] = square_class
            result_dict["probs"] = torch.sigmoid(logit).item()
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
