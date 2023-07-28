#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2022/9/8 15:32
@Message: null
"""
import random
import time
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage


class DataAugmenter(object):
    def __init__(self):
        super(DataAugmenter, self).__init__()

    @staticmethod
    def RandomAugmentation(img, corner_points):
        kps = KeypointsOnImage([
            Keypoint(x=corner_points[0], y=corner_points[1]),
            Keypoint(x=corner_points[2], y=corner_points[3]),
            Keypoint(x=corner_points[4], y=corner_points[5]),
            Keypoint(x=corner_points[6], y=corner_points[7]),
            Keypoint(x=corner_points[8], y=corner_points[9]),
            Keypoint(x=corner_points[10], y=corner_points[11]),
        ], shape=img.shape)

        # 数据增强2
        seq = iaa.SomeOf((1, 3), [
            # 1. 空操作
            iaa.Noop(),
            # 2. 调节亮度
            iaa.Multiply((1.2, 2)),  # 1-2 # 过亮
            iaa.Multiply((1.1, 1.8)),
            # 3. Gamma对比度,值越大越暗
            iaa.GammaContrast(gamma=(3.5, 3.8)),  # 过暗3.5-3.8
            iaa.GammaContrast(gamma=(1.2, 2.8)),
            # 4. 直方图均衡
            iaa.HistogramEqualization(),
            iaa.HistogramEqualization(),
            # 5. 色域变换
            iaa.WithColorspace(
                to_colorspace="HSV",
                from_colorspace="RGB",
                children=iaa.WithChannels(0, iaa.Add((0, 15)))),
            # 6. 调整对比度
            iaa.LinearContrast((0.75, 1.5)),
            # 7. 雨天模拟
            iaa.Rain(drop_size=(0.1, 0.30)),
            # 8. 对图片旋转 and 过近过远
            iaa.Affine(rotate=random.randint(-15, 15), scale=random.uniform(0.7, 1.1)),
            iaa.Affine(rotate=random.randint(-6, 8), scale=random.uniform(0.8, 1.2)),
            iaa.Affine(rotate=random.randint(-20, 20), scale=random.uniform(0.8, 1.2)), # add
            # 9. 随机裁剪 过远过近
            iaa.CropAndPad(percent=(-0.2, 0.3)),  # -0.2 0.3
            iaa.CropAndPad(percent=(-0.12, 0.24)),
            iaa.CropAndPad(percent=(0.1, 0.3)),  # add
            iaa.CropAndPad(percent=(-0.2, 0)),  # add
            # 10. 透视变换
            iaa.PerspectiveTransform(scale=(0, 0.12)),
            # 11. 均值偏移模糊
            iaa.MeanShiftBlur(spatial_radius=1),
            # 12. 运动模糊
            iaa.MotionBlur(k=(3, 5), angle=[-90, 90]),
            iaa.MotionBlur(k=(5, 9), angle=[120, 270]),
        ], random_order=True)

        # # 数据增强2-1
        # seq = iaa.SomeOf((2, 4), [
        #     # 2. 调节亮度
        #     iaa.OneOf([
        #         iaa.Multiply((1.2, 2)), # 1-2 # 过亮
        #         iaa.Multiply((0.1, 0.5)),# 过暗
        #     ]),
        #     iaa.OneOf([
        #         iaa.AddToBrightness(add=(50, 90)),  # 过亮
        #         iaa.AddToBrightness(add=(-100, -50)),  # 过暗
        #     ]),
        #     # 3. Gamma对比度,值越大越暗
        #     iaa.OneOf([
        #         iaa.GammaContrast(gamma=(3.5, 3.8)),    # 过暗3.5-3.8
        #         iaa.GammaContrast(gamma=(1.2, 2.8)),
        #     ]),
        #     iaa.OneOf([
        #         # 8. 对图片旋转 and 过近过远
        #         iaa.Affine(rotate=random.randint(-15, 15), scale=random.uniform(0.7, 1.1)),
        #         iaa.Affine(rotate=random.randint(-6, 8), scale=random.uniform(0.8, 1.2)),
        #     ]),
        #     iaa.OneOf([
        #         # 9. 随机裁剪 过远过近
        #         iaa.CropAndPad(percent=(-0.2, 0.3)),  # -0.2 0.3
        #         iaa.CropAndPad(),
        #     ]),
        #     # 12. 运动模糊
        #     iaa.OneOf([
        #         iaa.MotionBlur(k=(3, 5), angle=[-90, 90]),
        #         iaa.MotionBlur(k=(5, 9), angle=[120, 270]),
        #     ]),
        #     # 10. 透视变换
        #     iaa.PerspectiveTransform(scale=(0, 0.12)),
        #     # 4. 直方图均衡
        #     iaa.HistogramEqualization(),
        #     iaa.SomeOf((1, 2), [
        #         # 5. 色域变换
        #         iaa.WithColorspace(
        #             to_colorspace="HSV",
        #             from_colorspace="RGB",
        #             children=iaa.WithChannels(0, iaa.Add((0, 15)))),
        #         # 6. 调整对比度
        #         iaa.LinearContrast((0.5, 3)),
        #         # 7. 雨天模拟
        #         iaa.Rain(drop_size=(0.1, 0.30)),
        #     ]),
        #     # 11. 均值偏移模糊
        #     iaa.SomeOf((1, 2), [
        #         iaa.MeanShiftBlur(spatial_radius=(2, 8), color_radius=(23, 24)),
        #         iaa.BilateralBlur(),
        #         iaa.AverageBlur(),
        #         iaa.AverageBlur(k=(5, 11)),
        #         iaa.GaussianBlur(),
        #         iaa.GaussianBlur(sigma=(1, 5)),
        #     ]),
        #     iaa.Fliplr(),
        # ], random_order=True)

        #         # 增强3 一塌糊涂
        #         seq = iaa.SomeOf((2, 3), [
        #             # 1.极端光照
        #             iaa.OneOf([
        #                 # 强光照
        #                 iaa.OneOf([
        #                     iaa.Multiply((1.2, 1.8)),  # 1-2 # 过亮
        #                     iaa.GammaContrast(gamma=(0.3, 0.6)),  # 稍亮
        #                     iaa.AddToBrightness(add=(50, 90)),  # 过亮
        #                     iaa.LinearContrast(alpha=(1.5, 3.)),  # 亮遮挡
        #                 ]),
        #                 # 落光照
        #                 iaa.OneOf([
        #                     iaa.Multiply((0.1, 0.5)),  # 过暗
        #                     iaa.GammaContrast(gamma=(1.8, 4.8)),  # 过暗3.5-3.8
        #                     iaa.AddToBrightness(add=(-100, -50)),  # 过暗
        #                     iaa.LinearContrast(alpha=(1.5, 3.)),  # 亮遮挡
        #                     iaa.LinearContrast(alpha=(0.5, 0.8)),  # 暗遮挡
        #                 ]),
        #             ]),

        #             # 2.仿射，透视变换
        #             iaa.OneOf([
        #                 iaa.Affine(rotate=random.randint(-15, 15), scale=random.uniform(0.7, 1.1)),
        #                 iaa.Affine(rotate=random.randint(-6, 8), scale=random.uniform(0.8, 1.2)),
        #                 iaa.Affine(),
        #                 iaa.PerspectiveTransform(scale=(0, 0.12)),
        #             ]),
        #             # 3.随机裁剪 过远过近
        #             iaa.OneOf([
        #                 iaa.CropAndPad(percent=(-0.2, 0.3)),  # -0.2 0.3
        #                 iaa.CropAndPad(percent=(-0.12, 0.24)),
        #                 iaa.CropAndPad()
        #             ]),

        #             # 4.运动模糊
        #             iaa.OneOf([
        #                 iaa.MotionBlur(k=(3, 5), angle=[-90, 90]),
        #                 iaa.MotionBlur(k=(5, 9), angle=[120, 270]),
        #                 iaa.MotionBlur()
        #             ]),
        #             # 5.模糊
        #             iaa.OneOf([
        #                 iaa.MeanShiftBlur(spatial_radius=(2, 8), color_radius=(23, 24)),
        #                 iaa.BilateralBlur(),
        #                 iaa.AverageBlur(),
        #                 iaa.AverageBlur(k=(5, 11)),
        #                 iaa.GaussianBlur(),
        #                 iaa.GaussianBlur(sigma=(1, 5)),
        #             ]),
        #             # 6杂项
        #             iaa.OneOf([
        #                 # 直方图均衡
        #                 iaa.Sometimes(0.2, iaa.HistogramEqualization()),  # 和光纤对比度差不多
        #                 # 雨天模拟
        #                 iaa.Sometimes(0.2, iaa.Rain(drop_size=(0.1, 0.30))),
        #                 iaa.Fliplr()
        #             ]),
        #         ], random_order=True)

        # Augment keypoints and images.
        image_aug, kps_aug = seq(image=img, keypoints=kps)
        labels = []
        # use after.x_int and after.y_int to get rounded integer coordinates
        for k_i in range(len(kps.keypoints)):
            labels.append(kps_aug.keypoints[k_i].x)
            labels.append(kps_aug.keypoints[k_i].y)

        return image_aug, labels

    @staticmethod
    def TestAugmentation(img, corner_points):
        kps = KeypointsOnImage([
            Keypoint(x=corner_points[0], y=corner_points[1]),
            Keypoint(x=corner_points[2], y=corner_points[3]),
            Keypoint(x=corner_points[4], y=corner_points[5]),
            Keypoint(x=corner_points[6], y=corner_points[7]),
            Keypoint(x=corner_points[8], y=corner_points[9]),
            Keypoint(x=corner_points[10], y=corner_points[11]),
        ], shape=img.shape)
        seq = iaa.SomeOf(1, [
            # iaa.Fliplr(),
            # iaa.PerspectiveTransform(scale=(0, 0.2)),
            # iaa.CropAndPad(percent=(-0.2, 0.3)),
            # iaa.Affine(rotate=15, scale=(0.9, 0.9)),
            # iaa.MeanShiftBlur(spatial_radius=1),  # 均值偏移模糊
            # iaa.LinearContrast((0.75, 1.5)),  # 调整对比度
            # iaa.Multiply((1.2, 1.5)),
            # iaa.contrast.AllChannelsHistogramEqualization(),
            # iaa.HistogramEqualization(),

            # iaa.GammaContrast(gamma=(3.5, 3.8)),    # 过暗3.5-3.8
            # iaa.GammaContrast(gamma=(0.6, 1.2)),    # 过亮 0.6 不太好
            # iaa.MeanShiftBlur(spatial_radius=1),    # ok

            # iaa.Affine(rotate=random.randint(-15, 15), scale=random.uniform(0.7, 0.9)),
            # iaa.Affine(rotate=random.randint(-6, 8), scale=random.uniform(0.65, 1.2)),
            # iaa.Multiply((1.2, 2)), # 1-2 # 过亮
            # iaa.Multiply((1.1, 1.8)),
            # 5. 色域变换
            # iaa.WithColorspace(
            #     to_colorspace="HSV",
            #     from_colorspace="RGB",
            #     # children=iaa.WithChannels(0, iaa.Add((17, 18)))),
            #     children=iaa.WithChannels(0, iaa.Add((2, 10)))),
            # 11. 直方图均衡
            # iaa.HistogramEqualization(),
            # iaa.HistogramEqualization(),
            # 9. 调整对比度
            # iaa.LinearContrast((0.75, 1.5)),
            # iaa.LinearContrast(1.5),
            # 4. 随机裁剪
            iaa.CropAndPad(percent=(-0.2, 0.3)),  # -0.2 0.3
            # iaa.CropAndPad(percent=(-0.12, 0.24)),
            # iaa.MotionBlur(k=(3, 5), angle=[-45, 51]),
            # iaa.MotionBlur(k=(5, 9), angle=[120, 270]),
            # 透视变换
            # iaa.PerspectiveTransform(scale=(0, 0.15)),
            # iaa.PerspectiveTransform(scale=0.1),
            # iaa.LinearContrast(alpha=(1.4)),
            # iaa.Rotate((-100, -80)),
            # iaa.Rotate((80, 100))

        ], random_order=True)

        image_aug, kps_aug = seq(image=img, keypoints=kps)

        labels = []
        for k_i in range(len(kps.keypoints)):
            labels.append(kps_aug.keypoints[k_i].x)
            labels.append(kps_aug.keypoints[k_i].y)

        return image_aug, labels

    # 将CCPD中的文件名标签解码成坐标
    @staticmethod
    def decode_filename_label(img_path):
        label_points = []
        temp = img_path.split('-')[3].split('_')
        label_points.append(eval(temp[2].split('&')[0]))
        label_points.append(eval(temp[2].split('&')[1]))
        label_points.append(eval(temp[0].split('&')[0]))
        label_points.append(eval(temp[0].split('&')[1]))
        # four corner points
        label_points.append(eval(temp[2].split('&')[0]))
        label_points.append(eval(temp[2].split('&')[1]))
        label_points.append(eval(temp[3].split('&')[0]))
        label_points.append(eval(temp[3].split('&')[1]))
        label_points.append(eval(temp[0].split('&')[0]))
        label_points.append(eval(temp[0].split('&')[1]))
        label_points.append(eval(temp[1].split('&')[0]))
        label_points.append(eval(temp[1].split('&')[1]))
        return label_points

    # @staticmethod
    # def decode_filename_label_jx(img_path):


if __name__ == '__main__':
    files = os.listdir('../data/onlyone')
    file_name = '../data/onlyone/' + files[0]

    # image = ia.quokka(size=(256, 256))
    # image = Image.open(file_name)
    image = cv2.imread(file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image)

    # corners = [458, 394, 304, 357, 458, 394, 308, 410, 304, 357, 454, 341]
    # corners = [181,326,369,359,165,323,362,335,375,403,178,391]
    corners = DataAugmenter.decode_filename_label(file_name)
    start_time = time.time()

    for i in range(1):
        # img_after, label = DataAugmenter.RandomAugmentation(image, corners)
        img_after, label = DataAugmenter.TestAugmentation(image, corners)

    end_time = time.time() - start_time
    print(end_time)

    print(label)

    cv2.polylines(image, np.array(corners[4:]).reshape((1, 4, 2)), 1, color=(255, 0, 0), thickness=2)
    cv2.polylines(img_after, np.array(label[4:], dtype=np.int32).reshape((1, 4, 2)), 1, color=(255, 0, 0), thickness=2)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    plt.imshow(img_after)

    plt.show()
