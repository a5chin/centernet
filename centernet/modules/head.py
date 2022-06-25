from typing import Dict

import torch
from torch import nn


class CenterNetHead(nn.Module):
    def __init__(
        self, in_channels: int = 64, feat_channels: int = 64, num_classes=4
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_classes = num_classes

        self.heatmap_head = CenterNetHead._build_head(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=num_classes,
        )
        self.wh_head = CenterNetHead._build_head(
            in_channels=in_channels, feat_channels=feat_channels, out_channels=2
        )
        self.offset_head = CenterNetHead._build_head(
            in_channels=in_channels, feat_channels=feat_channels, out_channels=2
        )

    @staticmethod
    def _build_head(in_channels, feat_channels, out_channels) -> nn.Module:
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, out_channels, kernel_size=1),
        )
        return layer

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        heatmap = self.heatmap_head(x)
        wh = self.wh_head(x)
        offset = self.offset_head(x)

        return {"heatmap": heatmap, "wh": wh, "offset": offset}
