#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2022/11/18 23:15
@Message: null
"""
from .src_head import SRCHead

__all__ = ['build_head']
support_head = ['SRCHead']


def build_head(head_name, **kwargs):
    assert head_name in support_head, f'all support head is {support_head}'
    head = eval(head_name)(**kwargs)
    return head
