from typing import Dict

import torch
from torch import nn

from .backbone import resnet18
from .losses import GaussianFocalLoss, L1Loss
from .modules import CenterNetHead, CTResNetNeck
from .utils import gaussian_radius, gen_gaussian_target


class CenterNet(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.backbone(x)
        x = self.neck(x)
        feature = self.bbox_head(x)

        return feature

    def loss(
        self,
        feature: Dict[str, torch.Tensor],
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        imgs_shape: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        target = self.get_targets(
            gt_bboxes, gt_labels, feature["heatmap"].shape, imgs_shape
        )

        loss_center_heatmap = self.loss_center_haetmap(
            feature["heatmap"], target["center_heatmap_target"]
        ).sum(dim=(3, 2, 1))
        loss_wh = self.loss_wh(feature["wh"], target["wh_target"]).sum(dim=(3, 2, 1))
        loss_offset = self.loss_offset(
            feature["offset"],
            target["offset_target"],
        ).sum(dim=(3, 2, 1))

        return {
            "loss_center_heatmap": loss_center_heatmap.mean(),
            "loss_wh": loss_wh.mean(),
            "loss_offset": loss_offset.mean(),
        }

    def get_targets(
        self, gt_bboxes, gt_labels, feat_shape, imgs_shape
    ) -> Dict[str, torch.Tensor]:
        """Compute regression and classification targets in multiple images.
        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (list[int]): feature map shape with value [B, _, H, W]
            imgs_shape (list[int]): image shape in [h, w] format.
        Returns:
            tuple[dict,float]: The float value is mean avg_factor, the dict has
                components below:
                - center_heatmap_target (Tensor): targets of center heatmap, \
                    shape (B, num_classes, H, W).
                - wh_target (Tensor): targets of wh predict, shape \
                    (B, 2, H, W).
                - offset_target (Tensor): targets of offset predict, shape \
                    (B, 2, H, W).
                - wh_offset_target_weight (Tensor): weights of wh and offset \
                    predict, shape (B, 2, H, W).
        """
        bs, _, feat_h, feat_w = feat_shape

        height_ratio = feat_h / imgs_shape[:, 1]
        width_ratio = feat_w / imgs_shape[:, 2]

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w]
        )
        wh_target = gt_bboxes[:, -1].new_zeros([bs, 2, feat_h, feat_w])
        offset_target = gt_bboxes[:, -1].new_zeros([bs, 2, feat_h, feat_w])
        wh_offset_target_weight = gt_bboxes[:, -1].new_zeros([bs, 2, feat_h, feat_w])

        center_x = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) * width_ratio / 2
        center_y = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) * height_ratio / 2
        gt_centers = torch.cat([center_x.unsqueeze(1), center_y.unsqueeze(1)], dim=1)

        for j, ct in enumerate(gt_centers):
            ctx_int, cty_int = ct.int()
            ctx, cty = ct
            scale_box_h = (gt_bboxes[j, 3] - gt_bboxes[j, 1]) * height_ratio[j]
            scale_box_w = (gt_bboxes[j, 2] - gt_bboxes[j, 0]) * width_ratio[j]
            radius = gaussian_radius([scale_box_h, scale_box_w], min_overlap=0.3)
            radius = max(0, int(radius))
            ind = gt_labels[j] - 1
            gen_gaussian_target(
                center_heatmap_target[j, ind, :, :], [ctx_int, cty_int], radius
            )

            wh_target[j, 0, cty_int, ctx_int] = scale_box_w
            wh_target[j, 1, cty_int, ctx_int] = scale_box_h

            offset_target[j, 0, cty_int, ctx_int] = ctx - ctx_int
            offset_target[j, 1, cty_int, ctx_int] = cty - cty_int

            wh_offset_target_weight[j, :, cty_int, ctx_int] = 1

        return {
            "center_heatmap_target": center_heatmap_target,
            "wh_target": wh_target,
            "offset_target": offset_target,
        }
