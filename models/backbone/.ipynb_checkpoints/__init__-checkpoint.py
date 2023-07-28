#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2022/11/18 23:06
@Message: null
"""

from .resnet import *
__all__ = ['build_backbone']

support_backbone = ['resnet18', 'deformable_resnet18', 'resnet34',
                    'resnet50', 'deformable_resnet50',
                    'resnet101', 'resnet152']


def build_backbone(backbone_name, **kwargs):
    assert backbone_name in support_backbone, f'all support backbone is {support_backbone}'
    backbone = eval(backbone_name)(**kwargs)
    return backbone
