from typing import List, Tuple

import numpy as np
import torch
import wandb
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torchvision import transforms

from ..data.utils import grid_to_squares
from .models import UNet

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
        mode: str = "density",
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
            mode (str): for which model Looper works 'density' or 'classifier'

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
        self.mode = mode

    def run(self) -> float | Tuple[float]:
        """Run a single epoch loop.

        Returns:
            Mean absolute error.
        """
        if self.mode == "classifier":
            return self._run_classifier()

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

        for i, (image, label, n_points, _) in enumerate(self.loader):
            # move images and labels to given device
            if self.transforms:
                image = self.transforms(image)
            image = image.to(self.device)
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

    def freeze_layers(self, mode: str) -> None:
        if isinstance(self.network, torch.nn.DataParallel):
            self.network.module.freeze_layers(mode)
        else:
            self.network.freeze_layers(mode)

    def _run_classifier(self) -> float:
        # reset current results and add next entry for running loss
        self.true_values = []
        self.predicted_values = []
        self.running_loss.append(0)

        self.network.train(not self.validation)

        for i, (image, image_class, _) in enumerate(self.loader):
            # move images and images classes to given device
            if self.transforms:
                image = self.transforms(image)
            image = image.to(self.device)
            image_class = image_class.to(self.device)

            # clear accumulated gradient if in train mode
            if not self.validation:
                self.optimizer.zero_grad()

            # get classifier prediction
            result = self.network(image)

            # calculate loss and update running loss
            loss = self.loss(result, image_class)

            self.running_loss[-1] += image.shape[0] * loss.item() / self.size

            # update weights if in train mode
            if not self.validation:
                loss.backward()
                self.optimizer.step()

            # loop over batch samples
            for true, predicted in zip(image_class, result):
                # update current epoch results
                self.true_values.append(true.item())
                self.predicted_values.append(predicted.item())

        # calculate errors and standard deviation
        if self.mode == "classifier":
            self.predicted_values = self.to_binary(np.array(self.predicted_values))
        self.update_errors()

        # print epoch summary
        self.log()
        if self.mode == "classifier":
            return self.f1
        if self.relative_error:
            return self.mean_abs_rel_err, self.mean_abs_err
        return self.mean_abs_err

    def sigmoid(self, x: NDArray) -> NDArray:
        return 1 / (1 + np.exp(-x))

    def to_binary(self, logits: NDArray, threshold=0.5) -> List[float]:
        probabilities = self.sigmoid(logits)
        return (probabilities > threshold).astype(float).tolist()

    def update_errors(self):
        """
        Calculate errors and standard deviation based on current
        true and predicted values.
        """
        if self.mode == "classifier":
            self.accuracy = accuracy_score(self.true_values, self.predicted_values)
            self.precision = precision_score(self.true_values, self.predicted_values)
            self.recall = recall_score(self.true_values, self.predicted_values)
            self.f1 = f1_score(self.true_values, self.predicted_values)
            self.roc_auc = roc_auc_score(self.true_values, self.predicted_values)
            self.confusion = confusion_matrix(self.true_values, self.predicted_values)
            stage = "train" if not self.validation else "val"
            metrics = {
                f"{stage}/loss": self.running_loss[-1],
                f"{stage}/accuracy": self.accuracy,
                f"{stage}/precision": self.precision,
                f"{stage}/recall": self.recall,
                f"{stage}/f1": self.f1,
                f"{stage}/confusion": self.confusion,
            }

        elif self.mode == "density":
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
        if self.mode == "density":
            print(
                f"{'Train' if not self.validation else 'Valid'}:\n"
                f"\tAverage loss: {self.running_loss[-1]:3.4f}\n"
                f"\tMean error: {self.mean_err:3.3f}\n"
                f"\tMean absolute error: {self.mean_abs_err:3.3f}\n"
                f"\tMean absolute relative error: {self.mean_abs_rel_err:1.4f}\n"
                f"\tError deviation: {self.std:3.3f}\n"
                f"\tMean square error: {self.mean_square_error:3.3f}"
            )
        elif self.mode == "classifier":
            print(
                f"{'Train' if not self.validation else 'Valid'}:\n"
                f"\tAverage loss: {self.running_loss[-1]:3.4f}\n"
                f"\tAccuracy: {self.accuracy:3.3f}\n"
                f"\tPrecision: {self.precision:3.3f}\n"
                f"\tRecall: {self.recall:3.3f}\n"
                f"\tF1: {self.f1:3.3f}\n"
                f"\tConfusion matrix:\n{self.confusion}"
            )


class Config(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")


def logit_to_class(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Calculate class values from logits input

    Args:
        logits (torch.Tensor): logits after classifier
        threshold (float, optional): _description_. Defaults to 0.5.

    Returns:
        torch.Tensor: _description_
    """
    probabilities = torch.sigmoid(logits)
    return (probabilities > threshold).double()


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


def get_concentration_factor(
    virus_type: str, concentration_factor: float | None = None
) -> float:
    """Calculate the concentration factor for a given virus type.

    Args:
        virus_type (str): The type of the virus, expected to be "T4", "T7", or "T7c".
        concentration_factor (float, optional): The predefined concentration factor.
            If None, a default value is assigned based on the virus type.
            Defaults to None.

    Returns:
        float: The concentration factor for the specified virus type.

    Raises:
        ValueError: If the virus type is not one of the specified types (T4, T7, T7c),
            or if the virus type is not provided when concentration_factor is None.

    Note:
        The concentration factor for "T7c" should be updated after the release of T7c
            research.
    """

    # Mapping of virus types to their respective concentration factors
    concentration_factors = {
        "T4": 1.1,
        "T7": 1.0,
        "T7c": 1.0,
    }  # Update for T7c after research release

    # Assign default concentration factor if not provided
    if concentration_factor is None:
        if virus_type in concentration_factors:
            concentration_factor = concentration_factors[virus_type]
        else:
            raise ValueError("Wrong type of virus. Should be T4, T7 or T7c")
    elif virus_type not in concentration_factors:
        raise ValueError("Wrong type of virus. Should be T4, T7 or T7c")

    return concentration_factor
