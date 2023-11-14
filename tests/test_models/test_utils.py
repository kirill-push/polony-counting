from unittest.mock import patch

from models.utils import predict


def test_predict_single_file() -> None:
    # Mock os.path.isdir to return False, indicating that the path is a file
    with patch("os.path.isdir", return_value=False):
        # Mock predict_one_image to return a predefined dictionary
        with patch(
            "models.utils.predict_one_image",
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
                "models.utils.predict_one_image",
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
