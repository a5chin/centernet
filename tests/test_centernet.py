import sys
from pathlib import Path

import torch

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir) + "/../")

from centernet import CenterNet


def test_model():
    images = torch.randn(16, 3, 512, 512).clamp(min=0.0)
    gt_bboxes = torch.Tensor([0.0260, 0.5886, 2.1809, 1.9412]).repeat(16, 1)
    gt_labels = torch.randint(1, 5, (16,))
    imgs_shape = torch.randint(500, 1000, (16, 3))

    model = CenterNet(num_classes=4)

    out = model(images, gt_bboxes, gt_labels, imgs_shape)

    assert isinstance(out, dict)
