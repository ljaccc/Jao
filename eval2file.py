#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2022/10/3 11:50
@Message: 评估模型
"""
import os
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import zipfile

from models.LPCDet import LPCDet
from utils.configure import configs
from utils.dataloader import LPDataset, lpcdet_dataset_collate
from utils.decode import decode_corner
from utils.util import DataProcessor, FileProcessor, FeatureDecoder


def lpcdnet_result2zip(file_dir, zip_path):
    for file in tqdm(os.listdir(file_dir)):
        file_path = os.path.join(file_dir, file)

        zip_file = zipfile.ZipFile(zip_path, 'a', zipfile.ZIP_DEFLATED)
        zip_file.write(file_path, file)
        zip_file.close()

# img_dir = r'F:\LPDate\CCPD2019\ccpd_tilt'
img_dir = 'data/onlyone'
gt_path = 'zip_file/gt'
det_path = 'zip_file/det'

if __name__ == '__main__':
    print('LPCNet eval to file')
    threshold = 0.5
    count_acc = 0
    weight_path = 'utils/w_SRC.pth'
    # 推理设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.get_device_name(device))

    # 创建模型并加载权重
    model_configs = {
        'backbone': {'type': 'resnet50', 'pretrained': True, "in_channels": 3},
        'neck': {'type': 'FPEM_FFM', 'fpem_repeat': 2},  # 特征融合，FPN or FPEM_FFM
        'head': {'type': 'SRCHead', 'num_classes': 1},
    }

    model = LPCDet(model_config=model_configs)
    model.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=True)
    model.to(device)
    model.eval()

    start_time = time.time()
    count = 0
    for file_name in tqdm(os.listdir(img_dir)):

        file_path = os.path.join(img_dir, file_name)
        # 读取图片
        image = Image.open(file_path)
        image_np = np.array(image)
        image_shape = image_np.shape
        # gt
        img_name = file_path.split('\\')[-1]
        box_corner = DataProcessor.decode_label_jxlpd(img_name)
        # box_corner = DataProcessor.decode_label(img_name)
        gt = box_corner[:, 4:-1]
        gt_label = np.expand_dims(box_corner[:, 4:-1].reshape(4, 2), 0)
        #
        image_data = DataProcessor.resize_image(image, [512, 512])
        # 对图片进行归一化和变换通道 (C, W, H)
        image_data = np.expand_dims(
            np.transpose(DataProcessor.preprocess_input(np.array(image_data, dtype=np.float32)), (2, 0, 1)), 0)

        # 模型推理
        with torch.no_grad():
            img_input = torch.from_numpy(image_data).type(torch.FloatTensor).to(device)
            # 模型推理
            box_heatmap, corner_heatmap, corner_offset, corner_point = model(img_input)

        # 特征解码
        # SRC
        outputs = FeatureDecoder.decode_corner(corner_heatmap, corner_offset)
        results = DataProcessor.postprocess_corner(outputs, image_shape=[image_shape[0], image_shape[1]])

        IoU = DataProcessor.bbox_iou_eval(gt_label[0], results)
        count += 1

        gt_corner = '%d,%d,%d,%d,%d,%d,%d,%d,plate\n' % tuple(gt[0])
        pre_corner = '%d,%d,%d,%d,%d,%d,%d,%d\n' % tuple(results[0].reshape(8))

        if IoU > threshold and DataProcessor.is_clockwise(results[0]):
            count_acc += 1
        else:
            pre_corner = ''
        # 将读取的gt和预测结果分别写入txt文件
        FileProcessor.lpcdnet_result(count, gt_corner, pre_corner, gt_path, det_path)


    # 将txt文件压缩
    dir_path = 'zip_file'
    for file_name in os.listdir(dir_path):
        file_path = dir_path + '/' + file_name
        zip_path = file_path + '.zip'
        print(file_path)
        print(zip_path)
        assert not os.path.exists(
            zip_path), 'Error: the file path "%s" is exists, append write is unnecessary' % zip_path
        lpcdnet_result2zip(file_path, zip_path)


    # eval_acc = round(count_acc / len(dataset), 3) * 100
    time_spend = time.time() - start_time
    # print('model accuracy is {:.3f}%.'.format(eval_acc))
    print('Eval complete in {:.0f}m {:.0f}s.'.format(time_spend//60, time_spend % 60))
