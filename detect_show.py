#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2023/04/24 22:45
@Message: null
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models.LPCDet import LPCDet
from utils.util import DataProcessor, FeatureDecoder, DataVisualizer


def run(opts, model, device):
    # 加载模型权重
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(opts.weight))

    count_acc = 0

    save_path = opts.save_path
    count = len(os.listdir(opts.image_dir))
    print(count)
    for file_name in tqdm(os.listdir(opts.image_dir)):
        file_path = os.path.join(opts.image_dir, file_name)
        # 读取图片
        image = Image.open(file_path)
        image_np = np.array(image)
        #############
        image_shape = image_np.shape
        # # gt
        img_name = file_path.split('\\')[-1]
        # # for jxlpd
        # box_corner = DataProcessor.decode_label_jxlpd(img_name)
        # #  for ccpd
        # # box_corner = DataProcessor.decode_label(img_name)
        # gt_label = np.expand_dims(box_corner[:, 4:-1].reshape(4, 2), 0)
        #
        image_data = DataProcessor.resize_image(image, opts.input_size, undistorted=opts.undistorted)
        # 对图片进行归一化和变换通道 (C, W, H)
        image_data = np.expand_dims(np.transpose(DataProcessor.preprocess_input(np.array(image_data, dtype=np.float32)), (2, 0, 1)), 0)    # (1,3,512,512)

        # 模型推理
        with torch.no_grad():
            img_input = torch.from_numpy(image_data).type(torch.FloatTensor).to(device)
            # 模型推理
            box_heatmap, corner_heatmap, corner_offset, corner_point = model(img_input)

        # 特征解码
        # SRC
        outputs = FeatureDecoder.decode_corner(corner_heatmap, corner_offset)
        results = DataProcessor.postprocess_corner(outputs, image_shape=[image_shape[0], image_shape[1]], undistorted=opts.undistorted)
        # print(results)
        res = np.array(results[0], dtype=np.int32).tolist()
        # print(res)
        # 展示检测结果，默认False
        if opts.show:
            # 计算检测的IoU
            # IoU = DataProcessor.bbox_iou_eval(gt_label[0], results)
            # print(IoU)
            # if IoU > opts.threshold:
            #     count_acc += 1
            # else:
            img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            # 绿色 (0, 255, 0)
            # 红色 (0, 0, 255)
            # 蓝色 (255, 255, 0)
            # 黄色 (0, 255, 255)
            # DataVisualizer.draw_box(img, gt_label, color=(0, 0, 255), save=False)
            DataVisualizer.draw_box(img, results, color=(0, 255, 255), save=False)
            # print(save_path+'/'+img_name)
            DataVisualizer.draw_box(img, results, color=(0, 255, 0), save_path=save_path+'/'+img_name)

        # return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default=r'F:\6Model_deployment\LPCDet_L\weight\LPCDet_epoch19_acc0.931034_loss0.2453.pth', help='weight of LPCDet_L')
    parser.add_argument('--threshold', default=0.9, help='IoU')
    parser.add_argument('--image_dir', type=str, default=r'F:\9lp_data\select_520x304', help='input image')
    parser.add_argument('--save_path', type=str, default='output_one', help='input image')
    parser.add_argument('--input_size', default=[512, 512], help='input image size')
    parser.add_argument('--undistorted', default=True, help='input image size')
    parser.add_argument('--show', default=True, help='are you show?')
    opt = parser.parse_args()
    print(opt)

    model_configs = {
        'backbone': {'type': 'resnet50', 'pretrained': False, "in_channels": 3},
        'neck': {'type': 'FPEM_FFM', 'fpem_repeat': 2},  # 特征融合，FPN or FPEM_FFM
        'head': {'type': 'SRCHead', 'num_classes': 1},
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LPCDet(model_config=model_configs)

    res = run(opt, model, device)

    print(res)


