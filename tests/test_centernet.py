import sys
from pathlib import Path

import torch

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from centernet import CenterNet


def test_model():
    images = torch.randn(16, 3, 512, 512).clamp(min=0.0)
    gt_bboxes = torch.Tensor([0.0260, 0.5886, 2.1809, 1.9412]).repeat(16, 1)
    gt_labels = torch.randint(1, 5, (16,))
    imgs_shape = torch.randint(500, 1000, (16, 3))

    model = CenterNet(num_classes=4)

    feature = model(images)
    loss = model.loss(feature, gt_bboxes, gt_labels, imgs_shape)

    assert loss["loss_center_heatmap"] is not None
    assert loss["loss_wh"] is not None
    assert loss["loss_offset"] is not None

    det_bboxes, det_labels = model.get_bboxes(
        center_heatmap_preds=feature["heatmap"],
        wh_preds=feature["wh"],
        offset_preds=feature["offset"],
    )

    assert det_bboxes.shape[:-1] == det_labels.shape
