#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2022/7/26 20:00
@Message: loss for LPCDet_L
"""


import torch
import torch.nn.functional as F

from utils.util import FeatureDecoder


def focal_loss(pred, target):
    pred = pred.permute(0, 2, 3, 1)
    # -------------------------------------------------------------------------#
    #   找到每张图片的正样本和负样本
    #   一个真实框对应一个正样本
    #   除去正样本的特征点，其余为负样本
    # -------------------------------------------------------------------------#
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    # -------------------------------------------------------------------------#
    #   正样本特征点附近的负样本的权值更小一些
    # -------------------------------------------------------------------------#
    neg_weights = torch.pow(1 - target, 4)

    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    # -------------------------------------------------------------------------#
    #   计算focal loss。难分类样本权重大，易分类样本权重小。
    # -------------------------------------------------------------------------#
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    # -------------------------------------------------------------------------#
    #   进行损失的归一化
    # -------------------------------------------------------------------------#
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def reg_l1_loss(pred, target, mask):
    # --------------------------------#
    #   计算l1_loss
    # --------------------------------#
    pred = pred.permute(0, 2, 3, 1)
    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)

    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
#     loss = F.smooth_l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


def reg_l1_loss_c(pred, target, mask):
    # --------------------------------#
    #   计算l1_loss
    # --------------------------------#
    pred = pred.permute(0, 2, 3, 1)
    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 8)

    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
#     loss = F.smooth_l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


def calc_slack_loss(delta, alph=0.1, beta=5):
    c_loss = alph * delta * (torch.exp(delta / beta) - 1) if delta < beta else delta
    return c_loss

def calc_sl1_loss(delta, beta=5):
    sl1_loss = 0.5 * torch.pow(delta, 2) if torch.abs(delta) < beta else torch.abs(delta) - 0.5
    return sl1_loss


def slack_loss(pred_chm, pred_co, pred_bhm, pred_points):
    loss = 1e-5
    pred_bhm = FeatureDecoder.pool_nms(pred_bhm)
    pred_chm = FeatureDecoder.pool_nms(pred_chm)
    batch, c, h, w = pred_bhm.shape
    for b in range(batch):
        # box2corner
        box_heatmap = pred_bhm[b].permute(1, 2, 0).view([-1, c])  # (16384,1)
        corner_point = pred_points[b].permute(1, 2, 0).view([-1, 8])  # (16384, 8)
        # corner
        corner_heatmap = pred_chm[b].permute(1, 2, 0).view([-1, 4])
        corner_offset = pred_co[b].permute(1, 2, 0).view([-1, 8])

        center_conf, center_pred = torch.max(box_heatmap[..., 0], dim=-1)
        x_center = center_pred % 128
        y_center = center_pred / 128

        for corner_i in range(4):
            corner_conf, corner_pred = torch.max(corner_heatmap[:, corner_i], dim=-1)
            x_offset = corner_offset[corner_pred, corner_i * 2]
            y_offset = corner_offset[corner_pred, corner_i * 2 + 1]
            # 每个角点中心+对应的偏移量
            x_corner = corner_pred % 128 + x_offset
            y_corner = corner_pred / 128 + y_offset

            x_point = corner_point[center_pred, corner_i * 2] + x_center
            y_point = corner_point[center_pred, corner_i * 2 + 1] + y_center
            manhattan_distance = torch.abs(x_corner - x_point) + torch.abs(y_corner - y_point)
            # Slack loss
            loss += calc_slack_loss(manhattan_distance)
            ## mht loss
#             loss += manhattan_distance
            # smooth l1
#             loss += calc_sl1_loss(manhattan_distance)

    loss = loss / batch

    return loss


if __name__ == '__main__':
    print('utils_loss test')
    # pre_hm = torch.randn(1, 128, 128, 1)
    # label_hm = torch.randn(1, 128, 128, 1)
