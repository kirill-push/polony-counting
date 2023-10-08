import unittest

from models.utils import predict


class TestPredict(unittest.TestCase):
    def test_prediction(self):
        test_img = "data/raw/test_img.tif"
        state_dict = "models/polony_49_1.7496.pth"
        prediction = predict(
            path=test_img,
            path_to_model=state_dict,
        )
        return prediction[0].keys()


if __name__ == "__main__":
    unittest.main()
