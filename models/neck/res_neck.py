#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2022/11/19 18:46
@Message: null
"""
import torch
import torch.nn as nn


class ResNeck(nn.Module):
    def __init__(self, inplanes=2048, bn_momentum=0.1):
        super(ResNeck, self).__init__()
        self.bn_momentum = bn_momentum
        self.inplanes = inplanes
        self.deconv_with_bias = False

        # ----------------------------------------------------------#
        #   16,16,2048 -> 32,32,256 -> 64,64,128 -> 128,128,64
        #   利用ConvTranspose2d进行上采样。
        #   每次特征层的宽高变为原来的两倍。
        # ----------------------------------------------------------#
        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,
            num_filters=[256, 128, 64],
            num_kernels=[4, 4, 4],
        )

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            planes = num_filters[i]

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        c2, c3, c4, c5 = x
        return self.deconv_layers(c5)


if __name__ == '__main__':
    print()
    p2 = torch.randn((1, 256, 128, 128))
    p3 = torch.randn((1, 512, 64, 64))
    p4 = torch.randn((1, 1024, 32, 32))
    p5 = torch.randn((1, 2048, 16, 16))
    # inputs = torch.randn((1, 3, 512, 512))

    inputs = [p2, p3, p4, p5]
    model = ResNeck()
    outputs = model(inputs)
    print(outputs.shape)

    print(f'The parameters size of model is '
          f'{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0} MB')
