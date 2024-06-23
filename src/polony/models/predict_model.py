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
from typing import Dict

import pandas as pd
import torch
from torchvision import transforms

from ..data.utils import read_tiff
from .models import Classifier, UNet
from .utils import get_concentration_factor, logit_to_class

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
parser.add_argument(
    "--output_file",
    "-o",
    type=str,
    help="Path to save results, should be csv file",
)
parser.add_argument(
    "--virus_type",
    "-v",
    type=str,
    help="Can be T4, T7 or T7c",
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
) -> Dict[str, Dict[int, int]]:
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
        Dict[str, Dict[int, int]]: prediction - dict with name of image as key, and
            dictionaries as values. One value dict for one image;
            value dict = {square_id: number of points}
    """
    if channels not in [1, 2]:
        raise ValueError("Not correct value of channels, must be 1 or 2")
    is_dir = os.path.isdir(path)
    predictions = dict()
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
            predictions[image_path] = predict_one_image(
                image_path,
                density,
                classifier,
                channels,
                device,
                classifier_threshold,
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


def save_predictions_to_csv(
    predictions: Dict[str, Dict[int, int]],
    output_file: str,
    virus_type: str,
    concentration_factor: float | None = None,
    sample_volume_per_slide: float = 5.0,
    field: float = 2.9,
    grid: float = 0.1,
) -> None:
    """Save the predictions in a CSV file with a specified format and calculations.

    Args:
        predictions (Dict[str, Dict[int, int]]): Dictionary with predictions.
            Key - video name, value - dictionary {frame_number: prediction}.
        output_file (str): Path to the output CSV file.
        virus_type (str): Type of virus. Can be T4, T7 or T7c.
        concentration_factor (float | None): Dilution values for each point. If None,
            than use standart value.
            Defaults to None.
        sample_volume_per_slide (float): The sample volume per slide.
            Defaults to 5.0.
        field (float): Size of sample field.
            Defauts to 2.9.
        grid (float): Size of grid on sample field.
            Defaults to 0.1.
    """
    # Define concentration factor
    concentration_factor = get_concentration_factor(
        virus_type=virus_type,
        concentration_factor=concentration_factor,
    )
    # Create a list to hold all rows
    rows = []
    wg_area = field / grid
    # Iterate over each prediction
    for image_name, squares in predictions.items():
        # Initialize an empty list for the square values
        square_values = [squares.get(i, "") for i in range(1, 25)]

        # Calculate the average number of polonies/grid
        non_empty_values = [val for val in square_values if val != ""]
        average_polonies_per_grid = (
            sum(non_empty_values) / len(non_empty_values) if non_empty_values else 0
        )

        # Calculate fage_abundance
        fage_abundance = (
            1000 * wg_area * average_polonies_per_grid / sample_volume_per_slide
        )
        fage_abundance *= concentration_factor

        # Create the row
        row = [
            virus_type,
            image_name,
            "",
            "",
            concentration_factor,
            sample_volume_per_slide,
            field,
            grid,
            fage_abundance,
        ] + square_values
        rows.append(row)

    # Create a DataFrame
    column_names = [
        "TYPE",
        "slide ID",
        "SAMPLING DATE",
        "POLONY DATE",
        "CONCENTRATION FACTOR",
        "SAMPLE VOLUM[ul]",
        "field [Pixel^2]",
        "grid [Pixel^2]",
        "FAGE ABUNDENCE [fage mL-1]",
    ] + [f"field {i}" for i in range(1, 25)]
    df = pd.DataFrame(rows, columns=column_names)

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)


def main(args):
    predictions = predict(
        path=args.path,
        path_to_model=args.path_to_model,
    )
    if args.virus_type == "T4":
        concentration_factor = 1.1
    elif args.virus_type == "T7":
        concentration_factor = 1.0
    elif args.virus_type == "T7c":
        concentration_factor = 1.0  # Need to change after release of T7c research
    else:
        raise ValueError("Virus type should be T4, T7 or T7c")
    save_predictions_to_csv(
        predictions=predictions,
        output_file=args.output_file,
        virus_type=args.virus_type,
        concentration_factor=concentration_factor,
        sample_volume_per_slide=5.0,
        field=2.9,
        grid=0.1,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
