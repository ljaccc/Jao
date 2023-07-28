#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2022/10/3 11:50
@Message: null
"""

import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.LPCDet import LPCDet
from utils.configure import configs
from utils.dataloader import LPDataset, lpcdet_dataset_collate
from utils.util import DataProcessor, FeatureDecoder

if __name__ == '__main__':
    print('eval dataloader')
    threshold = 0.8
    count_acc = 0
    weight_path = 'utils/SRC.pth'
    # 推理设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.get_device_name(device))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>> LPCDet_L <<<<<<<<<<<<<<<<<<<<<<<<")
    print(weight_path)
    print("IoU threshold = ", threshold)
    # 创建模型并加载权重
    model = LPCDet(model_config=configs.model_configs).to(device)
    model.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=True)
    model.eval()
    # 加载数据集
    start_time = time.time()
    img_dir = 'data/one2'
    print(img_dir)
    dataset = LPDataset(img_dir, input_shape=(512, 512), num_classes=1, train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=6, collate_fn=lpcdet_dataset_collate)
    print('data length is {}'.format(len(dataset)))
    for data in tqdm(dataloader):
        batch_images, batch_box_hms, batch_corner_point, batch_corner_hms, batch_corner_regs, batch_corner_mask, gt = data
        batch_images = batch_images.to(device)
        batch_corner_hms = batch_corner_hms.to(device)
        batch_corner_regs = batch_corner_regs.to(device)
        # batch_box_hms = batch_box_hms.to(device)
        # batch_corner_point = batch_corner_point.to(device)

        with torch.set_grad_enabled(False):
            box_heatmap, corner_heatmap, corner_offset, corner_point = model(batch_images)
            # 计算检测准确率
            # CETCO
            de_result = FeatureDecoder.decode_corner_by_center(box_heatmap, corner_point)
            de_result = DataProcessor.postprocess_corner(de_result, undistorted=True)
            # CDM
            det_result = FeatureDecoder.decode_corner(corner_heatmap, corner_offset)
            det_result = DataProcessor.postprocess_corner(det_result)

            print("de_result:")
            print(de_result)

            print("det_result:")
            print(det_result)

            for bs in range(batch_corner_hms.shape[0]):
                IoU = DataProcessor.bbox_iou_eval(gt[bs], de_result[bs])
                # print('IoU:', IoU)
                if IoU > threshold:
                    count_acc += 1

    eval_acc = round(count_acc / len(dataset), 3) * 100
    time_spend = time.time() - start_time
    print('model accuracy is {:.3f}%.'.format(eval_acc))
    print('Eval complete in {:.0f}m {:.0f}s.'.format(time_spend//60, time_spend % 60))
