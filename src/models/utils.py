import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import wandb
from torchvision import transforms

from data.utils import grid_to_squares, read_tiff
from models.models import UNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FIRST_HORIZONTAL = 158
FIRST_VERTICAL = 158
SQUARE_SIZE = [316, 316]
# [vertical lines, horizontal lines] = [8, 5] by ImageJ
NUMBER_OF_LINES = [8, 5]

MEAN, STD = ([12.69365111, 2.47628206], [13.35308926, 2.45260453])
normalize = transforms.Normalize(MEAN, STD)


class Looper:
    """Looper handles epoch loops, logging, and plotting."""

    def __init__(
        self,
        network: torch.nn.Module,
        device: torch.device,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: torch.utils.data.DataLoader,
        dataset_size: int,
        validation: bool = False,
        regressor=None,
        batch_idx=0,
        relative_error=False,
        wandb_bool=False,
        transforms=None,
    ):
        """
        Initialize Looper.

        Args:
            network: already initialized model
            device: a device model is working on
            loss: the cost function
            optimizer: already initialized optimizer link to network parameters
            data_loader: already initialized data loader
            dataset_size: no. of samples in dataset
            plot: matplotlib axes
            validation: flag to set train or eval mode
            regressor: None or model for counting objects from network output

        """
        self.network = network
        self.device = device
        self.loss = loss
        self.optimizer = optimizer
        self.loader = data_loader
        self.size = dataset_size
        self.validation = validation
        self.running_loss: List[float] = []
        self.regressor = regressor
        self.batch_idx = batch_idx
        self.relative_error = relative_error
        self.wandb_bool = wandb_bool
        self.transforms = transforms

    def run(self):
        """Run a single epoch loop.

        Returns:
            Mean absolute error.
        """
        # reset current results and add next entry for running loss
        self.true_values = []
        self.predicted_values = []
        self.running_loss.append(0)

        if self.regressor is None:
            # set a proper mode: train or eval
            self.network.train(not self.validation)
        else:
            self.regressor.train(not self.validation)
            self.network.train(False)

        for i, (image, label, n_points, path) in enumerate(self.loader):
            # move images and labels to given device
            image = self.transforms(image).to(self.device)
            label = label.to(self.device)
            n_points = n_points.to(self.device)

            # clear accumulated gradient if in train mode
            if not self.validation:
                self.optimizer.zero_grad()

            # get model prediction (a density map)
            result = self.network(image)

            if self.regressor is not None:
                result = self.regressor(result)

            # calculate loss and update running loss
            if self.regressor is None:
                loss = self.loss(result, label)
            else:
                loss = self.loss(result, n_points)
            self.running_loss[-1] += image.shape[0] * loss.item() / self.size

            # update weights if in train mode
            if not self.validation:
                loss.backward()
                self.optimizer.step()

            if self.regressor is None:
                # loop over batch samples
                for true, predicted in zip(label, result):
                    # integrate a density map to get no. of objects
                    # note: density maps were normalized to 100 * no.of objects
                    # to make network learn better
                    true_counts = torch.sum(true).item() / 100
                    predicted_counts = torch.sum(predicted).item() / 100

                    # update current epoch results
                    self.true_values.append(true_counts)
                    self.predicted_values.append(predicted_counts)

            else:
                # loop over batch samples
                for true_counts, predicted_counts in zip(n_points, result):
                    # update current epoch results
                    self.true_values.append(true_counts.item())
                    self.predicted_values.append(torch.round(predicted_counts).item())

        # calculate errors and standard deviation
        self.update_errors()

        # print epoch summary
        self.log()
        if self.relative_error:
            return self.mean_abs_rel_err, self.mean_abs_err
        return self.mean_abs_err

    def update_errors(self):
        """
        Calculate errors and standard deviation based on current
        true and predicted values.
        """
        self.err = [
            true - predicted
            for true, predicted in zip(self.true_values, self.predicted_values)
        ]
        self.relative_err = [
            (true - predicted) / true
            for true, predicted in zip(self.true_values, self.predicted_values)
        ]
        self.square_err = [error * error for error in self.err]
        self.abs_err = [abs(error) for error in self.err]
        self.abs_rel_err = [abs(error) for error in self.relative_err]
        self.mean_err = sum(self.err) / self.size
        self.mean_abs_err = sum(self.abs_err) / self.size
        self.mean_abs_rel_err = sum(self.abs_rel_err) / self.size
        self.mean_square_error = sum(self.square_err) / self.size
        self.std = np.array(self.err).std()

        stage = "train" if not self.validation else "val"
        metrics = {
            f"{stage}/loss": self.running_loss[-1],
            f"{stage}/mean_err": self.mean_err,
            f"{stage}/MAE": self.mean_abs_err,
            f"{stage}/MARE": self.mean_abs_rel_err,
            f"{stage}/std": self.std,
            f"{stage}/MSE": self.mean_square_error,
        }
        if self.wandb_bool:
            wandb.log(metrics)

    def log(self):
        """Print current epoch results."""
        print(
            f"{'Train' if not self.validation else 'Valid'}:\n"
            f"\tAverage loss: {self.running_loss[-1]:3.4f}\n"
            f"\tMean error: {self.mean_err:3.3f}\n"
            f"\tMean absolute error: {self.mean_abs_err:3.3f}\n"
            f"\tMean absolute relative error: {self.mean_abs_rel_err:1.4f}\n"
            f"\tError deviation: {self.std:3.3f}\n"
            f"\tMean square error: {self.mean_square_error:3.3f}"
        )


class Config(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{key}'"
        )


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
        total number of points in image
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
