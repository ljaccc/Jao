#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2022/10/3 11:50
@Message: eval for LPCDet_L
"""
import os
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
    threshold = 0.7
    count_acc = 0
    # ../lpdata/train_run/LPCDet_L/20230422-13_56/weight/LPCDet_last_acc0.999800_loss0.0895.pth
    weight_path = '../lpdata/train_run/LPCDet_L/20230424-14_24/weight/LPCDet_epoch66_acc1.000000_loss0.0670.pth'
    # 推理设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.get_device_name(device))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>> LPCDet_L <<<<<<<<<<<<<<<<<<<<<<<<")
    print(weight_path)
    print("IoU threshold = ", threshold)
    # 创建模型并加载权重
    model = LPCDet(model_config=configs.model_configs).to(device)
    model_name = model.get_model_name()
    model.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=True)
    model.eval()
    # 加载数据集
    start_time = time.time()
    img_dir = '/student5/yjf/mydata/CCPD2019'
    txt_paths = ['ccpd_rotate.txt', 'ccpd_tilt.txt', 'ccpd_blur.txt', 'ccpd_db.txt', 'ccpd_fn.txt',
                 'ccpd_challenge.txt']
#     txt_paths = ['train_256.txt', 'val_128.txt']
    # -----------------save eval info--------------------
    date_info = time.strftime("%Y%m%d-%H_%M", time.localtime())
    save_txt = "../lpdata/eval_logs/{}-{}.txt".format(model_name, date_info)

    with open(save_txt, 'a', encoding='utf8') as f:
        f.write(">>>>>>>>>>>>>>>>>>>>>>>>>>>>> %s Eval Result INFO<<<<<<<<<<<<<<<<<<<<<<<<\n\n" % model_name)
        f.write("Eval date: {}\nEval IoU threshold: {}\nEval dataset: {}\nEval weight{}\n\n"
                .format(date_info, threshold, img_dir, weight_path))

    for i_p in range(len(txt_paths)):
        count_acc = 0
        txt_path = '../lpdata/ccpd_txt'
        txt_path = os.path.join(txt_path, txt_paths[i_p])
        print('='*16, txt_path, '='*18)
        dataset = LPDataset(img_dir, txt_path, input_shape=(512, 512), num_classes=1, train=False)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=6, collate_fn=lpcdet_dataset_collate)
        print('data length is {}'.format(len(dataset)))
        with open(save_txt, 'a', encoding='utf8') as f:
            f.write("{} {}-{} {}\n".format('='*17, txt_path[:-4], len(dataset), '='*24))

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
                # de_result = decode_corner_(box_heatmap, box_offset, corner_point)
                # de_result = DataProcessor.postprocess_corner(de_result, undistorted=True)
                det_result = FeatureDecoder.decode_corner(corner_heatmap, corner_offset)
                det_result = DataProcessor.postprocess_corner(det_result)
                for bs in range(batch_corner_hms.shape[0]):
                    IoU = DataProcessor.bbox_iou_eval(gt[bs], det_result[bs])
                    # print('IoU:', IoU)
                    if IoU > threshold:
                        count_acc += 1

        eval_acc = round(count_acc / len(dataset), 3) * 100
        print('model accuracy is {:.3f}%.\n'.format(eval_acc))
        with open(save_txt, 'a', encoding='utf8') as f:
            f.write('model accuracy is {:.3f}%.\n\n'.format(eval_acc))

    time_spend = time.time() - start_time
    print('Eval complete in {:.0f}m {:.0f}s.'.format(time_spend//60, time_spend % 60))
    with open(save_txt, 'a', encoding='utf8') as f:
        f.write('Eval complete in {:.0f}m {:.0f}s.'.format(time_spend//60, time_spend % 60))
