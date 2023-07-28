#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2022/7/18 21:10
@Message: null
"""

import torch.nn as nn


class SRCHead(nn.Module):

    def __init__(self, inner_channel=64, out_channel=64, num_classes=1, init_weight=True):
        super(SRCHead, self).__init__()

        # license plate box heatmap
        self.box_hm = nn.Sequential(
            nn.Conv2d(inner_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, num_classes, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

        # license plate corner heatmap
        self.corner_hm = nn.Sequential(
            nn.Conv2d(inner_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, num_classes * 4, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        # corner heatmap offset
        self.corner_offset = nn.Sequential(
            nn.Conv2d(inner_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, 8, kernel_size=1, stride=1)
        )
        # corner point
        self.corner_point = nn.Sequential(
            nn.Conv2d(inner_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, 8, kernel_size=1, stride=1)
        )

    # 1,64,128,128
    def forward(self, features):
        # 1,64,128,128
        # box heatmap
        box_heatmap = self.box_hm(features)    # (1,1,128,128)
        # corner heatmap
        corner_heatmap = self.corner_hm(features)
        # corner offset
        corner_offset = self.corner_offset(features)  # (1,8,128,128)
        # corner point
        corner_point = self.corner_point(features)  # (1,8,128,128)

        return box_heatmap, corner_heatmap, corner_offset, corner_point


if __name__ == '__main__':
    print('CenterHead test')
    model = SRCHead(in_channel=512,
                    inner_channel=64)

    print(f'The parameters size of model is '
          f'{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0} MB')
    print('done')
