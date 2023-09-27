from typing import List

import numpy as np
import torch
import wandb


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
                    self.predicted_values.append(
                        torch.round(predicted_counts).item()
                    )

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
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )
