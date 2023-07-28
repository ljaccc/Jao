#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2022/7/31 12:43
@Message: null
"""

import random

import numpy as np
import math
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from utils.data_augmenter import DataAugmenter
from utils.util import DataProcessor, GaussianHeatmap


class LPDataset(Dataset):
    def __init__(self, img_dir, txt_path=None, input_shape=(512, 512), num_classes=1, train=True, undistorted=True):
        super(LPDataset, self).__init__()
        self.img_dir = img_dir
        self.img_paths = []

        if txt_path is None:
            # 读取文件夹内的文件路径
            for file in os.listdir(self.img_dir):
                self.img_paths.append(self.img_dir + '/' + file)
        else:
            with open(txt_path, 'r', encoding='utf-8') as f:
                txt_lines = f.readlines()
                for line in txt_lines:
                    self.img_paths.append(os.path.join(self.img_dir, line.strip('\n')))

        # if shuffle:  # 文件乱序
        #     random.shuffle(self.img_paths)

        self.input_shape = input_shape
        self.output_shape = (int(input_shape[0] / 4), int(input_shape[1] / 4))
        self.num_classes = num_classes
        self.train = train
        self.undistorted = undistorted

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        filename = self.img_paths[item]
        # 数据增强
        image, box_corner, gt_label = self.get_random_data(filename, self.input_shape, random=self.train)

        # lb = filename.split("_")[-7][-6:]

        # imgs = Image.open(filename)
        # show_images(imgs, gt_label)

        # box
        batch_box_hm = np.zeros((self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        # batch_box_mask = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)
        # corner points
        batch_corner_hm = np.zeros((self.output_shape[0], self.output_shape[1], 4), dtype=np.float32)  # 留个坑，只允许一个类
        batch_corner_reg = np.zeros((self.output_shape[0], self.output_shape[1], 8), dtype=np.float32)
        batch_corner_mask = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)
        batch_corner_point = np.zeros((self.output_shape[0], self.output_shape[1], 8), dtype=np.float32)

        # if len(box_corner) != 0:  # 将label 调整到输出尺寸
        labels = np.array(box_corner[:, :-1], dtype=np.float32)
        labels[:, [0, 2, 4, 6, 8, 10]] = np.clip(labels[:, [0, 2, 4, 6, 8, 10]] / self.input_shape[1] *
                                                 self.output_shape[1], 0, self.output_shape[1] - 1)
        labels[:, [1, 3, 5, 7, 9, 11]] = np.clip(labels[:, [1, 3, 5, 7, 9, 11]] / self.input_shape[0] *
                                                 self.output_shape[0], 0, self.output_shape[0] - 1)

        for i in range(len(box_corner)):
            tlabel = labels[i].copy()
            cls_id = int(box_corner[i, -1])
            h, w = tlabel[3] - tlabel[1], tlabel[2] - tlabel[0]
            if h > 0 and w > 0:
                # 计算box的热图半径
                radius = GaussianHeatmap.gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                # 计算box 的中心点
                ct = np.array([(tlabel[0] + tlabel[2]) / 2, (tlabel[1] + tlabel[3]) / 2], dtype=np.float32)
                # 对box的中心点取整，便于后续标签中hm 的偏移量
                ct_int = ct.astype(np.int32)
                # 绘制label 中box的高斯热图
                batch_box_hm[:, :, cls_id] = GaussianHeatmap.draw_gaussian(batch_box_hm[:, :, cls_id], ct_int, radius)

                # 将对应的mask 设置为1
                # batch_box_mask[ct_int[1], ct_int[0]] = 1
                # 计算corner 的热图半径
                radius_point = max(0, int(radius * 0.7))
                # 计算label 中的corner 热图中心偏移量
                # corner points 的标签
                for p in range(4):
                    # 计算corner point 的中心
                    cp = np.array([tlabel[4:][p * 2], tlabel[4:][p * 2 + 1]], dtype=np.float32)  # [57.5 39.25]
                    # 对corner point 的中心点（点坐标本身）取整
                    cp_int = cp.astype(np.int32)  # [57 39]
                    # 绘制label 中 corner point 的高斯热图
                    batch_corner_hm[:, :, p] = GaussianHeatmap.draw_gaussian(batch_corner_hm[:, :, p], cp_int,
                                                                             radius_point)  # ok
                    # 计算label 中的corner 热图中心偏移量
                    batch_corner_reg[cp_int[1], cp_int[0], p * 2] = cp[0] - cp_int[0]
                    batch_corner_reg[cp_int[1], cp_int[0], p * 2 + 1] = cp[1] - cp_int[1]
                    # 将对应的mask 设置为1
                    batch_corner_mask[cp_int[1], cp_int[0]] = 1
                    # 4 个corner对应值（8个），这里是corner坐标-box中心的差值
                    batch_corner_point[cp_int[1], cp_int[0], p * 2] = cp[0] - ct[0]
                    batch_corner_point[cp_int[1], cp_int[0], p * 2 + 1] = cp[1] - ct[1]

        # 对图片进行归一化和变换通道 (C, W, H)
        image = np.transpose(DataProcessor.preprocess_input(image), (2, 0, 1))
        return image, batch_box_hm, batch_corner_point, batch_corner_hm, batch_corner_reg, batch_corner_mask, gt_label

    def rand(self, a=0., b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, file_name, input_shape, random=False):
        # 读取图片并转换成RGB图像
        image = Image.open(file_name)
        # 获得预测框，车牌角点和已编码的车牌
        # img_name: 01-86_91-298&341_449&414-458&394_308&410_304&357_454&341-0_0_14_28_24_26_29-124-24.jpg
        # box_corner: [[298 341 449 414 304 357 454 341 458 394 308 410 0]]
        # lp_code: [12 41 55 35 31 33 36]
        img_name = file_name.split('/')[-1]
        # TODO: 嘉兴数据集和CCPD不一样的
        box_corner = DataProcessor.decode_label_jxlpd(img_name)
        gt_label = box_corner[:, 4:-1].copy()

        if random:
            corner = []
            for corners in box_corner:  # 这里有个bug，只能处理包含一个标签的数据
                for point in corners[:-1]:
                    corner.append(point)

            image = np.array(image)
            image, label = DataAugmenter.TestAugmentation(image, corner)
            label.append(box_corner[0][-1])
            image = Image.fromarray(image)
            box_corner = np.expand_dims(np.array(label), axis=0)

        image_data, box_corner = DataProcessor.resize_image_with_boxes(image, box_corner,
                                                                       size=input_shape, undistorted=self.undistorted)
        # show_image(image, box_corner[0])
        if type(image_data) is not np.ndarray:
            image_data = np.array(image_data, np.float32)

        return image_data, box_corner, gt_label  # box_corner: [[228 150 294 182 230 157 296 150 298 173 232 180 0]]


# DataLoader中collate_fn 使用
def lpcdet_dataset_collate(batch):
    imgs, batch_box_hms, batch_corner_points, batch_corner_hms, batch_corner_regs, batch_corner_masks = [], [], [], [], [], []
    batch_gts = []
    # batch_lbs = []
    for img, batch_box_hm, batch_corner_point, batch_corner_hm, batch_corner_reg, batch_corner_mask, batch_gt in batch:
        imgs.append(img)
        batch_box_hms.append(batch_box_hm)
        batch_corner_points.append(batch_corner_point)
        batch_corner_hms.append(batch_corner_hm)
        batch_corner_regs.append(batch_corner_reg)
        batch_corner_masks.append(batch_corner_mask)
        batch_gts.append(batch_gt)
        # batch_lbs.append(batch_lb)

    try:
        # 转换为Tensor类型
        batch_gts = torch.from_numpy(np.array(batch_gts)).type(torch.FloatTensor)
        batch_images = torch.from_numpy(np.array(imgs)).type(torch.FloatTensor)
        batch_box_hms = torch.from_numpy(np.array(batch_box_hms)).type(torch.FloatTensor)
        batch_corner_points = torch.from_numpy(np.array(batch_corner_points)).type(torch.FloatTensor)
        batch_corner_hms = torch.from_numpy(np.array(batch_corner_hms)).type(torch.FloatTensor)
        batch_corner_regs = torch.from_numpy(np.array(batch_corner_regs)).type(torch.FloatTensor)
        batch_corner_masks = torch.from_numpy(np.array(batch_corner_masks)).type(torch.FloatTensor)
    except:
        print("")

    return batch_images, batch_box_hms, batch_corner_points, batch_corner_hms, batch_corner_regs, batch_corner_masks, \
           batch_gts


if __name__ == '__main__':
    print('dataloader test')
    img_dir = '../data/ccpd_two'
    dataset = LPDataset(img_dir, input_shape=(512, 512), num_classes=1, train=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=lpcdet_dataset_collate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('data length is {}'.format(len(dataset)))
    for data in dataloader:
        bs_images, bs_box_hms, bs_corner_points, bs_corner_hms, bs_corner_regs, bs_corner_masks, gt = data
        bs_image = bs_images.to(device)
        bs_box_hm = bs_box_hms.to(device)
        bs_corner_point = bs_corner_points.to(device)
        bs_corner_hm = bs_corner_hms.to(device)
        bs_corner_reg = bs_corner_regs.to(device)
        bs_corner_reg_masks = bs_corner_masks.to(device)
        print()

    print('done')
