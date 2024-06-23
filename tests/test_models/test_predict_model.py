from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.polony.models.predict_model import predict


def test_predict_single_file() -> None:
    # Mock os.path.isdir to return False, indicating that the path is a file
    with patch("os.path.isdir", return_value=False):
        # Mock predict_one_image to return a predefined dictionary
        with patch(
            "src.polony.models.predict_model.predict_one_image",
            return_value={1: {"result": 10, "class": 1.0, "density": "mocked_density"}},
        ):
            # Call the predict function
            result = predict(path="mocked_path")

            # Check that the result matches expectations
            assert len(result) == 1
            assert result[0][1]["result"] == 10
            assert result[0][1]["class"] == 1
            assert result[0][1]["density"] == "mocked_density"


def test_predict_directory() -> None:
    # To test the directory scenario, mock os.path.isdir to return True
    with patch("os.path.isdir", return_value=True):
        # Mock os.listdir to return a list of files
        with patch("os.listdir", return_value=["image1", "image2"]):
            # Mock predict_one_image
            with patch(
                "src.polony.models.predict_model.predict_one_image",
                side_effect=[
                    {1: {"result": 10, "class": 1.0, "density": "density1"}},
                    {2: {"result": 20, "class": 1.0, "density": "density2"}},
                ],
            ):
                result = predict(path="mocked_directory_path")

                # Check that the result matches expectations
                assert len(result) == 2
                assert result[0][1]["result"] == 10
                assert result[0][1]["density"] == "density1"
                assert result[1][2]["result"] == 20
                assert result[1][2]["density"] == "density2"


@pytest.fixture
def mock_denisty_model():
    mock = MagicMock()
    mock_model_object = mock.return_value
    mock_model_object.forward.return_value = torch.randn(1, 1, 256, 256)
    mock_model_object.load_state_dict.return_value = mock_model_object
    mock_model_object.module = mock_model_object
    return mock


class FakeModel(torch.nn.Module):
    def forward(self, input):
        return torch.randn(1, 1, 256, 256)

    def load_state_dict(self, *args, **kwargs):
        pass


@patch("torch.nn.DataParallel")
def test_predict_one_image_real_work(mock_dataparallel):
    with patch("src.polony.models.predict_model.UNet", new=FakeModel) as mock_density:
        with patch(
            "src.polony.models.predict_model.Classifier", new=FakeModel
        ) as mock_classifier:
            mock_classifier.forward = lambda *args, **kwargs: torch.randn(1, 1)

            mock_dataparallel.side_effect = lambda model: model
            result = predict(
                "resources/raw/test/test_img.tif",
                density=mock_density(),
                classifier=mock_classifier(),
            )
            assert isinstance(result, List)
            assert isinstance(result[0], Dict)
from unittest.mock import patch, MagicMock, mock_open
from models.predict_model import main

from models.predict_model import parser
import tempfile


def test_argument_parsing():
    # Test with default arguments
    args = parser.parse_args([])
    assert args.path == "."
    assert args.path_to_model == "../models/polony_49_1.7496.pth"

    # Test with custom arguments
    custom_args = ["-p", "/custom/path", "-m", "/custom/model.pth"]
    args = parser.parse_args(custom_args)
    assert args.path == "/custom/path"
    assert args.path_to_model == "/custom/model.pth"


# Not ready
def not_ready_test_main_function_with_single_file():
    # Mock the parser arguments to simulate command line input
    args = MagicMock(path="mocked_path", path_to_model="mocked_model_path")

    # Mock os.path.isdir to return False, indicating that the path is a file
    with patch("os.path.isdir", return_value=False):
        # Mock predict_one_image to return a predefined dictionary
        with patch(
            "models.utils.predict_one_image",
            return_value={1: {"result": 10, "density": "mocked_density"}},
        ):
            # Mock the predict function
            with patch("models.utils.predict") as mock_predict:
                # Configure the mock predict function to use our mock predict_one_image
                mock_predict.return_value = [{1: {"result": 10, "density": "mocked_density"}}]

                # Mock the creation of a temporary directory
                with patch("tempfile.mkdtemp", return_value="/tmp/mock_temp_dir"):
                    # Mock the file writing operations
                    with patch("builtins.open", mock_open()):
                        with patch.object(tempfile, 'mkdtemp', return_value='/tmp/predictions'):
                            # Call the main function with the mocked arguments
                            main(args)

                            # Verify that the predict function was called with the correct arguments
                            mock_predict.assert_called_with("mocked_path", "mocked_model_path")

                            # TODO: Add assertions to check if the correct files are written
