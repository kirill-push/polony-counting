import os

import numpy as np
import torch
import wandb
import yaml
from torch import nn
from torchvision import transforms

from ..data.make_dataset import PolonyDataset
from ..data.utils import mean_std
from .models import Classifier, UNet
from .utils import Config, Looper

# folder to load config file
current_script_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_script_path)
CONFIG_PATH = os.path.join(root_path, "config", "config.yaml")

with open(CONFIG_PATH, "r") as file:
    config_yaml = yaml.load(file, Loader=yaml.FullLoader)
train_params = config_yaml["train"]


def train(
    dataset_name: str = train_params["dataset_name"],
    network_architecture: str | nn.Module = train_params["network_architecture"],
    learning_rate: float = train_params["learning_rate"],
    epochs: int = train_params["epochs"],
    batch_size: int = train_params["batch_size"],
    unet_filters: int = train_params["unet_filters"],
    convolutions: int = train_params["convolutions"],
    lr_patience: int = train_params["lr_patience"],
    input_channels: int = train_params["input_channels"],
    wandb_bool: bool = train_params["wandb_bool"],
    factor: float = train_params["factor"],
    res: bool = train_params["res"],
    loss: nn.MSELoss = nn.MSELoss(),
    freeze_threshold: int = 10,
) -> None:
    """Train chosen model on selected dataset.

    Args:
        dataset_name (str, optional): _description_.
            Defaults to train_params["dataset_name"].
        network_architecture (str | nn.Module, optional): _description_.
            Defaults to train_params["network_architecture"].
        learning_rate (float, optional): _description_.
            Defaults to train_params["learning_rate"].
        epochs (int, optional): _description_.
            Defaults to train_params["epochs"].
        batch_size (int, optional): _description_.
            Defaults to train_params["batch_size"].
        unet_filters (int, optional): _description_.
            Defaults to train_params["unet_filters"].
        convolutions (int, optional): _description_.
            Defaults to train_params["convolutions"].
        lr_patience (int, optional): _description_.
            Defaults to train_params["lr_patience"].
        input_channels (int, optional): _description_.
            Defaults to train_params["input_channels"].
        wandb_bool (bool, optional): _description_.
            Defaults to train_params["wandb_bool"].
        factor (float, optional): _description_.
            Defaults to train_params["factor"].
        res (bool, optional): _description_.
            Defaults to train_params["res"].
        loss (nn.MSELoss, optional): _description_.
            Defaults to nn.MSELoss().
        freeze_threshold (int, optional): _description_.
            Defaults to 10.
    """

    # use GPU if avilable
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # update params of training
    if network_architecture == "UNet":
        looper_mode = "density"
        lr_mode = "min"
    elif network_architecture == "Classifier":
        looper_mode = "classifier"
        lr_mode = "max"

    if wandb_bool:
        # start a new wandb run to track this script
        run = wandb.init(
            # set the wandb project where this run will be logged
            project="polony",
            save_code=True,
            # track hyperparameters and run metadata
            config={
                "learning_rate": learning_rate,
                "architecture": network_architecture,
                "dataset": dataset_name,
                "epochs": epochs,
                "lr_patience": lr_patience,
                "factor": factor,
            },
        )

        # Copy config to WandB
        config = wandb.config
    #         artifact = wandb.Artifact(name='neural_network', type='model')
    else:
        config = Config(
            {
                "learning_rate": learning_rate,
                "architecture": network_architecture,
                "dataset": dataset_name,
                "epochs": epochs,
                "lr_patience": lr_patience,
                "factor": factor,
            }
        )

    dataset = {}  # training and validation HDF5-based datasets
    dataloader = {}  # training and validation dataloaders
    shuffle = {"train": True, "valid": False}

    for mode in ["train", "valid"]:
        # expected HDF5 files in dataset_name/(train | valid).h5
        # turn on flips only for training dataset
        polony_dataset_params = (
            config_yaml["PolonyDataset_train"]
            if mode == "train"
            else config_yaml["PolonyDataset_val"]
        )
        dataset[mode] = PolonyDataset(**polony_dataset_params, mode=looper_mode)
        dataloader[mode] = torch.utils.data.DataLoader(
            dataset[mode], batch_size=batch_size, shuffle=shuffle[mode]
        )
    # initialize a model based on chosen network_architecture
    if network_architecture == "UNet":
        network = UNet(
            input_filters=input_channels,
            filters=unet_filters,
            N=convolutions,
            res=res,
        ).to(device)
    elif network_architecture == "Classifier":
        network = Classifier(in_channel=input_channels).to(device)
        loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(dataset["train"].pos_weight())
        )
    elif isinstance(network_architecture, nn.Module):
        network = network_architecture.to(device)
    else:
        raise ValueError(
            "Wrong network, shoud be nn.Module or string: 'Classifier' or 'UNet'"
        )
    network = torch.nn.DataParallel(network)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=config.learning_rate,
    )

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=lr_mode,
        patience=config.lr_patience,
        verbose=True,
        factor=config.factor,
    )

    MEAN, STD = mean_std(dataset["train"])
    normalize = transforms.Normalize(MEAN, STD)
    # create training and validation Loopers to handle a single epoch
    train_looper = Looper(
        network,
        device,
        loss,
        optimizer,
        dataloader["train"],
        len(dataset["train"]),
        wandb_bool=wandb_bool,
        transforms=normalize,
        mode=looper_mode,
    )
    # turn on freezing layers for better learning
    if looper_mode == "classifier":
        train_looper.freeze_layers("on")
        print("Layers were frozen")

    valid_looper = Looper(
        network,
        device,
        loss,
        optimizer,
        dataloader["valid"],
        len(dataset["valid"]),
        validation=True,
        wandb_bool=wandb_bool,
        transforms=normalize,
        mode=looper_mode,
    )

    # current best results (lowest mean absolute error on validation set)
    if lr_mode == "min":
        current_best = 100
        second_best = np.infty
    elif lr_mode == "max":
        current_best = 0
        second_best = -1
    for epoch in range(config.epochs):
        # turn off freezing layers after freeze_threshold epochs
        if epoch == freeze_threshold and looper_mode == "classifier":
            train_looper.freeze_layers("off")
            print("Layers were unfrozen")

        print(f"Epoch {epoch + 1}\n")

        # run training epoch and update learning rate
        train_looper.run()

        # run validation epoch
        with torch.no_grad():
            result = valid_looper.run()

        # update learning rate
        lr_scheduler.step(result)

        # update checkpoint if new best is reached
        filename = (
            f"{dataset_name}_"
            f"{network_architecture}_"
            f"{epoch}_"
            f"{result:.4f}.pth"
        )
        if lr_mode == "min":
            if result < current_best:
                current_best = result
                if result < 3:
                    torch.save(
                        network.state_dict(),
                        filename,
                    )

                print(f"\nNew best result: {result}")
            elif result <= second_best:
                second_best = result
                if result < 3:
                    torch.save(
                        network.state_dict(),
                        filename,
                    )
                print(f"\nNew best second result: {result}")
        elif lr_mode == "max":
            if result > current_best:
                current_best = result
                if result >= 0.7:
                    torch.save(
                        network.state_dict(),
                        filename,
                    )

                print(f"\nNew best result: {result}")
            elif result >= second_best:
                second_best = result
                if result >= 0.7:
                    torch.save(
                        network.state_dict(),
                        filename,
                    )
                print(f"\nNew best second result: {result}")
        print("\n", "-" * 80, "\n", sep="")

    print(f"[Training done] Best result: {current_best}")
    torch.save(network.state_dict(), f"{dataset_name}_last.pth")
    if wandb_bool:
        run.finish()


if __name__ == "__main__":
    print("Training parameters were taken from the config file")
    train(**train_params)
