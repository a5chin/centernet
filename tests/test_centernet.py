import sys
from pathlib import Path

import torch

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir) + "/../")

from centernet import CenterNet


def test_model():
    images = torch.randn(16, 3, 512, 512)
    gt_bboxes = torch.randn(16, 4)
    gt_labels = torch.randn(16)
    imgs_shape = torch.randint(500, 1000, (16, 3))

    model = CenterNet(num_classes=4)

    out = model(images, gt_bboxes, gt_labels, imgs_shape)

    assert isinstance(out, dict)
