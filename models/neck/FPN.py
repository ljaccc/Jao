#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2022/11/18 21:46
@Message: null
"""

import torch
import torch.nn.functional as F
from addict import Dict
from torch import nn

from models.backbone import build_backbone
from models.basic import ConvBnRelu


class FPN(nn.Module):
    def __init__(self, in_channels=None, inner_channels=512, **kwargs):
        """
        FPN
        Args:
            in_channels: 输入FPN中的通道数
            out_channels: FPN输出通道数
            inner_channels: 内部通道数 FPN原文分割网络设置为128
            **kwargs:
        """
        super().__init__()
        if in_channels is None:
            in_channels = [256, 512, 1024, 2048]
        inplace = True
        # feature 通道为64
        self.conv_out = inner_channels
        inner_channels = inner_channels // 4
        # reduce layers
        self.reduce_conv_c2 = ConvBnRelu(in_channels[0], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c3 = ConvBnRelu(in_channels[1], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c4 = ConvBnRelu(in_channels[2], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c5 = ConvBnRelu(in_channels[3], inner_channels, kernel_size=1, inplace=inplace)
        # Smooth layers
        self.smooth_p4 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p3 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p2 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)

    def forward(self, x):
        # c2: (1,256,128,128)
        # c3: (1,512,64,64)
        # c4: (1,1024,32,32)
        # c5: (1,2048,16,16)
        c2, c3, c4, c5 = x
        # Top-down
        p5 = self.reduce_conv_c5(c5)    # (1,64,16,16)
        p4 = self._upsample_add(p5, self.reduce_conv_c4(c4))    # (1,64,32,32)
        p4 = self.smooth_p4(p4)     # (1,16,32,32)
        p3 = self._upsample_add(p4, self.reduce_conv_c3(c3))    # (1,64,64,64)
        p3 = self.smooth_p3(p3)     # (1,16,64,64)
        p2 = self._upsample_add(p3, self.reduce_conv_c2(c2))    # (1,64,128,128)
        p2 = self.smooth_p2(p2)     # (1,64,128,128)

        x = self._upsample_cat(p2, p3, p4, p5)  # (1,256,128,128)
        return x

    def _upsample(self, x, size):
        y = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return y

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        y0 = self._upsample(x, size=(H, W))
        return y0 + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = self._upsample(p3, size=(h, w))
        p4 = self._upsample(p4, size=(h, w))
        p5 = self._upsample(p5, size=(h, w))
        return torch.cat([p2, p3, p4, p5], dim=1)


# class Net(nn.Module):
#     def __init__(self, model_config):
#         super(Net, self).__init__()
#         model_config = Dict(model_config)
#         backbone_type = model_config.backbone.pop('type')
#
#         self.backbone = build_backbone(backbone_type, **model_config.backbone)
#         self.neck = FPN(inner_channels=512)
#
#     def forward(self, x):
#         backbone_out = self.backbone(x)
#         neck_out = self.neck(backbone_out)
#         return neck_out


if __name__ == '__main__':
    print()
    model_configs = {
        'backbone': {'type': 'resnet50', 'pretrained': False, "in_channels": 3},
        'neck': {'type': 'FPN', 'inner_channels': 256},  # 特征融合，FPN or FPEM_FFM
        'head': {'type': 'CenterHead', 'num_classes': 1},
    }

    p2 = torch.randn((1, 256, 128, 128))
    p3 = torch.randn((1, 512, 64, 64))
    p4 = torch.randn((1, 1024, 32, 32))
    p5 = torch.randn((1, 2048, 16, 16))
    # inputs = torch.randn((1, 3, 512, 512))
    inputs = [p2, p3, p4, p5]
    model = FPN([256, 512, 1024, 2048], inner_channels=512)
    outputs = model(inputs)
    print(outputs.shape)

    print(f'The parameters size of model is '
          f'{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0} MB')
    print()
