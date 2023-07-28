#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2022/7/18 21:10
@Message: null
"""
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from addict import Dict

from models.backbone import build_backbone
from models.head import build_head
from models.neck import build_neck


class LPCDet(nn.Module):
    def __init__(self, model_config):
        super(LPCDet, self).__init__()
        model_config = Dict(model_config)
        backbone_type = model_config.backbone.pop('type')
        neck_type = model_config.neck.pop('type')
        head_type = model_config.head.pop('type')

        self.backbone = build_backbone(backbone_type, **model_config.backbone)
        self.neck = build_neck(neck_type, **model_config.neck)
        self.head = build_head(head_type, **model_config.head)

    def get_model_name(self):
        return self.__class__.__name__

    def forward(self, x):
        backbone_out = self.backbone(x)
        neck_out = self.neck(backbone_out)
        return self.head(neck_out)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_configs = {
        'backbone': {'type': 'resnet50', 'pretrained': False, "in_channels": 3},
        'neck': {'type': 'FPEM_FFM'},  # 特征融合，FPN or FPEM_FFM
        'head': {'type': 'SRCHead', 'num_classes': 1},
    }
    model = LPCDet(model_config=model_configs).to(device)
    print('%s test' % model.get_model_name())

    img = cv2.imread(
        '../data/base@0108620689655-91_81-130&434_310&494-321&499_132&499_119&433_308&433-0_0_8_27_25_24_4-115-19.jpg')
    img = cv2.resize(img, (512, 512))
    img = torch.unsqueeze(torch.Tensor(img), 1).permute(1, 3, 0, 2).contiguous()
    inputs = img.to(device)
    t_all = []
    for i in range(1):
        t1 = time.time()
        output = model(inputs)
        t2 = time.time()
        t_all.append(t2 - t1)

    print('average time:', np.mean(t_all) / 1)
    print('average fps:', 1 / np.mean(t_all))

    print('fastest time:', min(t_all) / 1)
    print('fastest fps:', 1 / min(t_all))

    print('slowest time:', max(t_all) / 1)
    print('slowest fps:', 1 / max(t_all))

    # print(model)
    print(f'The parameters size of model is '
          f'{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0} MB')

    print('done')




