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
