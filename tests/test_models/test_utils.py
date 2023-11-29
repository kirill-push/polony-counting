from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.polony.models.utils import Looper, predict


def test_predict_single_file() -> None:
    # Mock os.path.isdir to return False, indicating that the path is a file
    with patch("os.path.isdir", return_value=False):
        # Mock predict_one_image to return a predefined dictionary
        with patch(
            "src.polony.models.utils.predict_one_image",
            return_value={1: {"result": 10, "density": "mocked_density"}},
        ):
            # Call the predict function
            result = predict("mocked_path", "mocked_model_path")

            # Check that the result matches expectations
            assert len(result) == 1
            assert result[0][1]["result"] == 10
            assert result[0][1]["density"] == "mocked_density"


def test_predict_directory() -> None:
    # To test the directory scenario, mock os.path.isdir to return True
    with patch("os.path.isdir", return_value=True):
        # Mock os.listdir to return a list of files
        with patch("os.listdir", return_value=["image1", "image2"]):
            # Mock predict_one_image
            with patch(
                "src.polony.models.utils.predict_one_image",
                side_effect=[
                    {1: {"result": 10, "density": "density1"}},
                    {2: {"result": 20, "density": "density2"}},
                ],
            ):
                result = predict("mocked_directory_path", "mocked_model_path")

                # Check that the result matches expectations
                assert len(result) == 2
                assert result[0][1]["result"] == 10
                assert result[0][1]["density"] == "density1"
                assert result[1][2]["result"] == 20
                assert result[1][2]["density"] == "density2"


@pytest.fixture
def mock_network():
    return MagicMock(spec=torch.nn.Module)


@pytest.fixture
def mock_device():
    return torch.device("cpu")


@pytest.fixture
def mock_loss():
    return MagicMock(spec=torch.nn.MSELoss)


@pytest.fixture
def mock_optimizer():
    return MagicMock(spec=torch.optim.Optimizer)


@pytest.fixture
def mock_data_loader():
    return MagicMock(
        return_value=[
            (
                torch.randn(1, 2, 224, 224),
                torch.randn(1, 1, 224, 224),
                torch.tensor([1]),
                torch.tensor([1]),
                "path",
            )
        ]
    )


@pytest.fixture
def looper_instance(
    mock_network, mock_device, mock_loss, mock_optimizer, mock_data_loader
):
    return Looper(
        network=mock_network,
        device=mock_device,
        loss=mock_loss,
        optimizer=mock_optimizer,
        data_loader=mock_data_loader,
        dataset_size=1,
        validation=False,
    )


def test_looper_initialization(looper_instance):
    assert looper_instance is not None
    assert not looper_instance.validation


@pytest.mark.parametrize("validation", [True, False])
@pytest.mark.parametrize("regressor", [None, True])
def test_looper_run_method(looper_instance, mock_network, validation, regressor):
    looper_instance.validation = validation  # False
    if regressor is None:
        looper_instance.regressor = regressor  # None
    else:
        looper_instance.regressor = mock_network
    result = looper_instance.run()
    assert result is not None
    if regressor is None:
        assert looper_instance.network.train.called_with(not validation)
    else:
        assert looper_instance.regressor.train.called_with(not validation)


@pytest.mark.parametrize("wandb", [True, False])
def test_looper_update_errors_method(looper_instance, wandb):
    looper_instance.true_values = [1, 2, 3]
    looper_instance.predicted_values = [1, 1.5, 2.5]
    looper_instance.size = 3
    looper_instance.running_loss = [0.2]

    with patch("wandb.log") as mock_wandb_log:
        looper_instance.wandb_bool = wandb  # False
        looper_instance.update_errors()

        assert looper_instance.mean_err == pytest.approx((0 + 0.5 + 0.5) / 3)
        assert looper_instance.mean_abs_err == pytest.approx((0 + 0.5 + 0.5) / 3)
        assert looper_instance.mean_abs_rel_err == pytest.approx(
            (0 + 0.25 + 0.16667) / 3, abs=1e-4
        )
        assert looper_instance.std == pytest.approx(np.std([0, 0.5, 0.5]))
        if wandb:
            mock_wandb_log.assert_called_once()
        else:
            mock_wandb_log.assert_not_called()


@pytest.mark.parametrize("mode", ["density", "classifier"])
def test_looper_log_method(capsys, looper_instance, mode):
    looper_instance.mode = mode
    looper_instance.running_loss = [0.3]
    if mode == "density":
        looper_instance.mean_err = 0.1
        looper_instance.mean_abs_err = 0.2
        looper_instance.mean_abs_rel_err = 0.3
        looper_instance.std = 0.4
        looper_instance.mean_square_error = 0.5
    else:
        looper_instance.accuracy = 0.1
        looper_instance.precision = 0.2
        looper_instance.recall = 0.3
        looper_instance.f1 = 0.4
        looper_instance.confusion = 0.5

    looper_instance.log()

    captured = capsys.readouterr()
    assert "Average loss: 0.3000" in captured.out
    if mode == "density":
        assert "Mean error: 0.100" in captured.out
        assert "Mean absolute error: 0.200" in captured.out
        assert "Mean absolute relative error: 0.3000" in captured.out
        assert "Error deviation: 0.400" in captured.out
        assert "Mean square error: 0.500" in captured.out
    else:
        assert "Accuracy: 0.100" in captured.out
        assert "Precision: 0.200" in captured.out
        assert "Recall: 0.300" in captured.out
        assert "F1: 0.400" in captured.out
        assert "Confusion: 0.500" in captured.out
