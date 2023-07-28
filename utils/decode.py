#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2022/7/19 21:51
@Message: null
"""

import torch
from torch import nn


def pool_nms(heatmap, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def decode_corner(pre_hms, pre_offsets=None):
    # 热图的非极大值抑制，利用3×3 的卷积对热图进行最大值筛选，找出区域内得分最大的特征
    pre_hms = pool_nms(pre_hms)  # (1,4,128,128)
    # 这里应该是 bs, C, H, W，不太确定
    batch, c, output_h, output_w = pre_hms.shape
    detects = []
    for b in range(batch):
        # heatmap   (128,128,num_classes)   corner 的热图
        corner_heatmap = pre_hms[b].permute(1, 2, 0).view([-1, c])  # (16384,4)
        if pre_offsets is not None:
            pred_offsets = pre_offsets[b].permute(1, 2, 0).view([-1, 8])  # (16384,8)
        # corner 的4个点坐标，即每个corner_heatmap 上置信度最大的预测
        xy_list = []
        for corner_heatmap_i in range(4):
            corner_conf, corner_pred = torch.max(corner_heatmap[:, corner_heatmap_i], dim=-1)
            if pre_offsets is not None:
                x_offset = pred_offsets[corner_pred, corner_heatmap_i * 2]
                y_offset = pred_offsets[corner_pred, corner_heatmap_i * 2 + 1]
                # 每个角点中心+对应的偏移量
                x = corner_pred % 128 + x_offset
                y = corner_pred / 128 + y_offset
            else:
                x = corner_pred % 128
                y = corner_pred / 128
            xy_list.append([x, y])
        # 将其合并
        corners = torch.Tensor(xy_list)  # (4, 2)
        corners[:, [0]] /= output_w
        corners[:, [1]] /= output_h
        detects.append(corners)

    return detects


def decode_corner_(pred_bhm, pred_bo, pred_points):
    pred_bhm = pool_nms(pred_bhm)
    batch, c, h, w = pred_bhm.shape
    detects = []
    for b in range(batch):
        # box2corner
        box_heatmap = pred_bhm[b].permute(1, 2, 0).view([-1, c])  # (16384,1)
        box_offset = pred_bo[b].permute(1, 2, 0).view([-1, 2])
        corner_point = pred_points[b].permute(1, 2, 0).view([-1, 8])  # (16384, 8)
        center_conf, center_pred = torch.max(box_heatmap[..., 0], dim=-1)
        x_o = box_offset[center_pred, 0]
        y_o = box_offset[center_pred, 1]
        x_center = center_pred % 128 + x_o
        y_center = center_pred / 128 + y_o

        xy_list = []
        for corner_i in range(4):
            x_point = corner_point[center_pred, corner_i * 2] + x_center
            y_point = corner_point[center_pred, corner_i * 2 + 1] + y_center
            xy_list.append([x_point, y_point])

        corners = torch.Tensor(xy_list)  # (4, 2)
        corners[:, [0]] /= w
        corners[:, [1]] /= h
        detects.append(corners)
    return detects


# def cal_feature_IoU(gt_heatmap, gt_offset, det_heatmap, det_offset, undistorted=True):
#     # 根据gt和det 的特征层计算IoU
#     gt_result = decode_corner(gt_heatmap.permute(0, 3, 1, 2), gt_offset.permute(0, 3, 1, 2))
#     gt_result = postprocess_corner(gt_result, undistorted=undistorted)
#     det_result = decode_corner(det_heatmap, det_offset)
#     det_result = postprocess_corner(det_result, undistorted=undistorted)
#     return bbox_iou_eval(gt_result, det_result)


if __name__ == '__main__':
    print('utils_decode test')
