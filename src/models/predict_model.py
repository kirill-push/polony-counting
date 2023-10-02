import argparse
import os
import tempfile

from .models.predict_model import predict

# Configuring Argument parser
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# parser.add_argument(
#     "--download",
#     "-d",
#     type=bool,
#     default=True,
#     help="is it necessary ot not to download data from Google drive"
# )

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

ARGS = parser.parse_args()

if __name__ == "__main__":
    temp_folder = tempfile.mkdtemp()
    print("Creating temporary folder: {}".format(temp_folder))

    predictions = predict(
        path=ARGS.path,
        path_to_model=ARGS.path_to_model,
    )

    with open(os.path.join(temp_folder, "prediction"), "w") as file:
        for pred in predictions:
            for k, v in pred.items():
                file.write(f"Square {k}, number of points {v}\n")
