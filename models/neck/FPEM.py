#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2022/11/28 16:06
@Message: LPCDet_L
"""
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.basic import ConvBnRelu


class FPEM_FFM(nn.Module):
    def __init__(self, in_channels=None, inner_channels=128, out_channel=64, fpem_repeat=2, init_weight=True, **kwargs):
        """
        PANnet

        # debug: 梯度不能回传 https://github.com/NVlabs/FUNIT/issues/23
        :param in_channels: 基础网络输出的维度
        """
        super().__init__()
        if in_channels is None:
            in_channels = [256, 512, 1024, 2048]
        self.conv_out = inner_channels
        self.inplace = True
        # reduce layers
        self.reduce_conv_c2 = ConvBnRelu(in_channels[0], inner_channels, kernel_size=1, inplace=self.inplace)
        self.reduce_conv_c3 = ConvBnRelu(in_channels[1], inner_channels, kernel_size=1, inplace=self.inplace)
        self.reduce_conv_c4 = ConvBnRelu(in_channels[2], inner_channels, kernel_size=1, inplace=self.inplace)
        self.reduce_conv_c5 = ConvBnRelu(in_channels[3], inner_channels, kernel_size=1, inplace=self.inplace)
        self.fpems = nn.ModuleList()
        for i in range(fpem_repeat):
            self.fpems.append(FPEM(self.conv_out))
        self.out_channels = self.conv_out * 4
        self.adjust_conv = nn.Sequential(
            nn.Conv2d(self.out_channels, out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))


        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        c2, c3, c4, c5 = x
        # reduce channel
        c2 = self.reduce_conv_c2(c2)
        c3 = self.reduce_conv_c3(c3)
        c4 = self.reduce_conv_c4(c4)
        c5 = self.reduce_conv_c5(c5)

        c2_ffm, c3_ffm, c4_ffm, c5_ffm = torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.])
        # FPEM
        for i, fpem in enumerate(self.fpems):
            c2, c3, c4, c5 = fpem(c2, c3, c4, c5)
            if i == 0:
                c2_ffm = c2
                c3_ffm = c3
                c4_ffm = c4
                c5_ffm = c5
            else:
                c2_ffm += c2
                c3_ffm += c3
                c4_ffm += c4
                c5_ffm += c5

        # FFM
        c3 = F.interpolate(c3_ffm, c2_ffm.size()[-2:])  # (1,128,128,64)
        c4 = F.interpolate(c4_ffm, c2_ffm.size()[-2:])  # (1,128,32,32)
        c5 = F.interpolate(c5_ffm, c2_ffm.size()[-2:])  # (1,128,16,16)

        Fy = torch.cat([c2_ffm, c3, c4, c5], dim=1)  # (1,512,128,128)
        Fy = self.adjust_conv(Fy)
        return Fy


class FPEM(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        self.up_add1 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add2 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add3 = SeparableConv2d(in_channels, in_channels, 1)
        self.down_add1 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add2 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add3 = SeparableConv2d(in_channels, in_channels, 2)

    def forward(self, c2, c3, c4, c5):
        # up阶段
        c4 = self.up_add1(self._upsample_add(c5, c4))
        c3 = self.up_add2(self._upsample_add(c4, c3))
        c2 = self.up_add3(self._upsample_add(c3, c2))

        # down 阶段
        c3 = self.down_add1(self._upsample_add(c3, c2))
        c4 = self.down_add2(self._upsample_add(c4, c3))
        c5 = self.down_add3(self._upsample_add(c5, c4))
        return c2, c3, c4, c5

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:]) + y


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SeparableConv2d, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1,
                                        stride=stride, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.siLu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.siLu(x)
        return x


if __name__ == '__main__':
    p2 = torch.randn((1, 256, 128, 128))
    p3 = torch.randn((1, 512, 64, 64))
    p4 = torch.randn((1, 1024, 32, 32))
    p5 = torch.randn((1, 2048, 16, 16))
    # inputs = torch.randn((1, 3, 512, 512))
    inputs = [p2, p3, p4, p5]

    model = FPEM_FFM([256, 512, 1024, 2048],
                     inner_channels=128,
                     fpem_repeat=3)
    output = model(inputs)  # (1, 64, 128 ,128)
    print(output.shape)

    print(f'The parameters size of model is '
          f'{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0} MB')
