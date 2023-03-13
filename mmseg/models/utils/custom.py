import torch
from torch import nn

from mmcv.cnn import ConvModule


class SequenceConv(nn.ModuleList):
    """Sequence conv module.

    Args:
        in_channels (int): input tensor channel.
        out_channels (int): output tensor channel.
        kernel_size (int): convolution kernel size.
        sai_number (int): sequence length.
        conv_cfg (dict): convolution config dictionary.
        norm_cfg (dict): normalization config dictionary.
        act_cfg (dict): activation config dictionary.
    """

    def __init__(self, in_channels, out_channels, kernel_size, sai_number, conv_cfg, norm_cfg, act_cfg, stride=1):
        super(SequenceConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sai_number = sai_number
        self.stride = stride
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for _ in range(sai_number):
            self.append(
                ConvModule(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    stride = self.stride,
                    padding=self.kernel_size // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
            )

    def forward(self, sequence_imgs):
        """

        Args:
            sequence_imgs (Tensor): TxBxCxHxW

        Returns:
            sequence conv output: TxBxCxHxW
        """
        sequence_outs = []
        assert sequence_imgs.shape[0] == self.sai_number
        for i, sequence_conv in enumerate(self):
            sequence_out = sequence_conv(sequence_imgs[i, ...])
            sequence_out = sequence_out.unsqueeze(0)
            sequence_outs.append(sequence_out)

        sequence_outs = torch.cat(sequence_outs, dim=0)  # TxBxCxHxW
        return sequence_outs
