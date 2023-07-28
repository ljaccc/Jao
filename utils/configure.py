#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2022/7/19 20:23
@Message: config for LPCDet_L
"""

from easydict import EasyDict as edict

configs = edict()

# 模型参数
# deformable_resnet50, resnet50
configs['model_configs'] = {
        'backbone': {'type': 'resnet50', 'pretrained': True, "in_channels": 3},
        'neck': {'type': 'FPEM_FFM', 'fpem_repeat': 2},  # 特征融合，FPN or FPEM_FFM
        'head': {'type': 'SRCHead', 'num_classes': 1},
    }
configs['img_H'] = 512
configs['img_W'] = 512

# configs['image_shape'] = [304, 520]
# 手持设备的嘉兴数据
configs['image_shape'] = [1144, 1920]

# 训练参数
configs['train_dir'] = {'train': r'F:\6Model_deployment\LPCDet_L\data\val-jx',  # ../lpdata/ccpd_train
                        'val': r'F:\6Model_deployment\LPCDet_L\data\val-jx'}
# configs['train_data_path'] = '/student5/yjf/mydata/CCPD2019' # '/student5/yjf/mydata/CCPD2019'
# configs['train_txt_path'] = {'train': '../lpdata/ccpd_txt/train_small.txt',
#                              'val': '../lpdata/ccpd_txt/val_small.txt'}

configs['weight'] = r'F:\6Model_deployment\LPCDet_L\run\20230507-19_28\weight\LPCDet_best.pth'
configs['save_path'] = "run"
configs['seed'] = 42

configs['device'] = 'cuda:0'
configs['epochs'] = 50
configs['batch_size'] = 6
configs['num_workers'] = 4
configs['iou_threshold'] = 0.7
configs['num_classes'] = 1
configs['draw_epoch'] = 15
configs['calc_epoch'] = 2

# scale the loss
configs['scale'] = [4, 4, 2, 1, 1]

# 学习率优化器
configs['optimizer'] = 'Adam'
configs['lr_scheduler'] = 'MultLR'
configs['lr'] = 0.001
configs['lr_step'] = [20, 36]
# configs['lr_step'] = [50,70,90]
# configs['lr'] = 0.0002
# configs['lr_step'] = [25,28]
configs['lr_gamma'] = 0.2
configs['weight_decay'] = 2e-5
configs['step_size'] = 20
