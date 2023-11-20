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

from models.utils import predict

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
    default="../models/polony_49_1.7496.pth",
    help="Path to saved state dict",
)


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
