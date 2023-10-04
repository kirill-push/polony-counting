import unittest

import torch

from models.models import UNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TestUNet(unittest.TestCase):
    def test_forward_pass(self):
        nn = UNet(input_filters=2).to(device)
        input_data = torch.rand((1, 2, 316, 316)).to(device)
        output = nn(input_data)

        self.assertEqual(output.shape, (1, 1, 316, 316))


if __name__ == "__main__":
    unittest.main()
