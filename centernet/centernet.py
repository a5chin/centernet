from typing import Dict, Tuple

import torch
from torch import nn

from .backbone import resnet18
from .losses import GaussianFocalLoss, L1Loss
from .modules import CenterNetHead, CTResNetNeck
from .utils import (
    batched_nms,
    gaussian_radius,
    gen_gaussian_target,
    get_local_maximum,
    get_topk_from_heatmap,
    transpose_and_gather_feat,
)


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

    def get_bboxes(
        self,
        center_heatmap_preds,
        wh_preds,
        offset_preds,
        with_nms=False,
    ) -> Tuple[torch.Tensor]:
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): Center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): WH predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (list[Tensor]): Offset predicts for all levels
                with shape (B, 2, H, W).
            with_nms (bool): If True, do nms before return boxes.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds)

        bboxes, labels = [], []
        for img_id in range(len(center_heatmap_preds)):
            bbox, label = self._get_bboxes_single(
                center_heatmap_preds[img_id : img_id + 1],
                wh_preds[img_id : img_id + 1],
                offset_preds[img_id : img_id + 1],
                with_nms=with_nms,
            )
            bboxes.append(bbox)
            labels.append(label)

        return torch.stack(bboxes), torch.stack(labels)

    def _get_bboxes_single(
        self,
        center_heatmap_pred,
        wh_pred,
        offset_pred,
        with_nms=True,
    ) -> Tuple[torch.Tensor]:
        """Transform outputs of a single image into bbox results.

        Args:
            center_heatmap_pred (Tensor): Center heatmap for current level with
                shape (1, num_classes, H, W).
            wh_pred (Tensor): WH heatmap for current level with shape
                (1, num_classes, H, W).
            offset_pred (Tensor): Offset for current level with shape
                (1, corner_offset_channels, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor, Tensor]: The first item is an (n, 5) tensor, where
                5 represent (tl_x, tl_y, br_x, br_y, score) and the score
                between 0 and 1. The shape of the second tensor in the tuple
                is (n,), and each element represents the class label of the
                corresponding box.
        """
        batch_det_bboxes, batch_labels = self.decode_heatmap(
            center_heatmap_pred,
            wh_pred,
            offset_pred,
        )

        det_bboxes = batch_det_bboxes.view([-1, 5])
        det_labels = batch_labels.view(-1)

        # batch_border = det_bboxes.new_tensor(img_meta["border"])[..., [2, 0, 2, 0]]
        # det_bboxes[..., :4] -= batch_border

        if with_nms:
            det_bboxes, det_labels = self._bboxes_nms(det_bboxes, det_labels)

        return det_bboxes, det_labels

    def decode_heatmap(
        self,
        center_heatmap_pred: torch.Tensor,
        wh_pred: torch.Tensor,
        offset_pred: torch.Tensor,
        img_shape: Tuple[int] = (512, 512),
        k: int = 100,
        kernel: int = 3,
    ) -> Tuple[torch.Tensor]:
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
                shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            img_shape (Tuple[int]): image shape in [h, w] format.
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
                Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
                the following Tensors:

                - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
                - batch_topk_labels (Tensor): Categories of each box with \
                    shape (B, k)
        """
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]
        tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
        tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
        br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
        br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]), dim=-1)

        return batch_bboxes, batch_topk_labels + 1

    def _bboxes_nms(self, bboxes, labels, max_num: int = 100) -> Tuple[torch.Tensor]:
        if labels.numel() > 0:
            bboxes, keep = batched_nms(
                bboxes[:, :4], bboxes[:, -1].contiguous(), labels
            )
            if max_num > 0:
                bboxes = bboxes[:max_num]
                labels = labels[keep][:max_num]

        return bboxes, labels
