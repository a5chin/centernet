from torch import nn

from .conv_module import ConvModule


class CTResNetNeck(nn.Module):
    """The neck used in `CenterNet <https://arxiv.org/abs/1904.07850>`_ for
    object classification and box regression.

    Args:
        in_channels (int): Number of input channels.
        num_deconv_filters (tuple[int]): Number of filters per stage.
        num_deconv_kernels (tuple[int]): Number of kernels per stage.
        use_dcn (bool): If True, use DCNv2. Default: True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        in_channels,
        num_deconv_filters,
        num_deconv_kernels,
        use_dcn=True,
    ) -> None:
        super().__init__()
        assert len(num_deconv_filters) == len(num_deconv_kernels)
        self.use_dcn = use_dcn
        self.in_channels = in_channels
        self.deconv_layers = self._make_deconv_layer(
            num_deconv_filters, num_deconv_kernels
        )

    def _make_deconv_layer(self, num_deconv_filters, num_deconv_kernels):
        """use deconv layers to upsample backbone's output."""
        layers = []
        for i in range(len(num_deconv_filters)):
            feat_channels = num_deconv_filters[i]
            conv_module = ConvModule(
                in_channels=self.in_channels,
                out_channels=feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_fn="Conv2d",
                norm_fn="BatchNorm2d",
            )
            layers.append(conv_module)
            upsample_module = ConvModule(
                in_channels=feat_channels,
                out_channels=feat_channels,
                kernel_size=num_deconv_kernels[i],
                stride=2,
                padding=1,
                conv_fn="ConvTranspose2d",
                norm_fn="BatchNorm2d",
            )
            layers.append(upsample_module)
            self.in_channels = feat_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        outs = self.deconv_layers(x)
        return outs
