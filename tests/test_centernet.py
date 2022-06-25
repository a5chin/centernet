import sys
import warnings
from pathlib import Path

import torch

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir) + "/../")

from centernet import CenterNet


def test_model():
    images = torch.randn(4, 3, 255, 255)
    model = CenterNet()

    out = model(images)

    assert isinstance(out, dict)
