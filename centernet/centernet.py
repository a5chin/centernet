from typing import Dict

import torch
from torch import nn

from .backbone import resnet18
from .losses import GaussianFocalLoss, L1Loss
from .modules import CenterNetHead, CTResNetNeck
from .utils import gaussian_radius, gen_gaussian_target


class CenterNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = resnet18(pretrained=True)
        self.neck = CTResNetNeck(
            in_channels=512,
            num_deconv_filters=(256, 128, 64),
            num_deconv_kernels=(4, 4, 4),
        )
        self.bbox_head = CenterNetHead(
            in_channels=64, feat_channels=64, num_classes=num_classes
        )

        self.loss_center_haetmap = GaussianFocalLoss(loss_weight=1.0)
        self.loss_wh = L1Loss(loss_weight=0.1)
        self.loss_offset = L1Loss(loss_weight=1.0)

    def forward(
        self,
        x: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        imgs_shape,
    ) -> Dict[str, torch.Tensor]:
        x = self.backbone(x)
        x = self.neck(x)
        feature = self.bbox_head(x)

        target = self.get_targets(
            gt_bboxes, gt_labels, feature["heatmap"].shape, imgs_shape
        )

        loss_center_heatmap = self.loss_center_haetmap(
            feature["heatmap"], target["center_heatmat_target"]
        )
        loss_wh = self.loss_wh(
            feature["wh"], target["wh_target"], target["wh_offset_target_weight"]
        )
        loss_offset = self.loss_offset(
            feature["offset"],
            target["offset_target"],
            target["wh_offset_target_weight"],
        )

        return {
            "loss_center_heatmap": loss_center_heatmap,
            "loss_wh": loss_wh,
            "loss_offset": loss_offset,
        }
