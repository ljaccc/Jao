#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2022/11/18 23:07
@Message: null
"""

from .FPN import FPN
from .res_neck import ResNeck
from .FPEM import FPEM_FFM

__all__ = ['build_neck']
support_neck = ['FPN', 'ResNeck', 'FPEM_FFM']


def build_neck(neck_name, **kwargs):
    assert neck_name in support_neck, f'all support neck is {support_neck}'
    neck = eval(neck_name)(**kwargs)
    return neck
