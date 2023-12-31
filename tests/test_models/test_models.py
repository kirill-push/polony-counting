import pytest
import torch

from polony import Classifier, UNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_forward_pass():
    nn = UNet(input_filters=2).to(device).eval()
    input_data = torch.rand((1, 2, 316, 316)).to(device)
    output = nn(input_data)

    assert output.shape == (1, 1, 316, 316)


def test_forward_pass_with_res():
    nn = UNet(input_filters=2, res=True).to(device).eval()
    input_data = torch.rand((1, 2, 316, 316)).to(device)
    output = nn(input_data)

    assert output.shape == (1, 1, 316, 316)


def test_classifier():
    model = Classifier().to(device)
    inp = torch.randn(size=(4, 2, 256, 256)).to(device)
    model.freeze_layers("on")
    assert model(inp).shape == (4, 1)

    model.freeze_layers("off")
    assert model(inp).shape == (4, 1)

    with pytest.raises(ValueError):
        model.freeze_layers("onn")
