import torch
from models.models import UNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_forward_pass():
    nn = UNet(input_filters=2).to(device)
    input_data = torch.rand((1, 2, 316, 316)).to(device)
    output = nn(input_data)

    assert output.shape == (1, 1, 316, 316)
