#!/user/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from tqdm import tqdm
import torch
from torch.optim import lr_scheduler

from models.LPCDet import LPCDet
from utils.loss import focal_loss, reg_l1_loss_c, slack_loss
from utils.dataloader import LPDataset, lpcdet_dataset_collate
from torch.utils.data import DataLoader
from utils.configure import configs
from utils.util import DataVisualizer, DataProcessor, FeatureDecoder, UtilTools


def train(datasets, model, device):
    # 获取模型类名
    model_name = model.get_model_name()
    # 定义学习器
    if configs.optimizer == 'Adam':
        print(configs.optimizer)
        optimizer = torch.optim.Adam(
            [{'params': model.parameters(), 'lr': configs.lr, 'weight_decay': configs.weight_decay}])
    elif configs.optimizer == 'AdamW':
        print(configs.optimizer)
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay,
                                      amsgrad=True)
    else:
        optimizer = torch.optim.SGD(params=model.parameters(), lr=configs.lr, momentum=0.9,
                                    weight_decay=configs.weight_decay)
    # 学习率调节器
    if configs.lr_scheduler == 'Cosine':
        print(configs.lr_scheduler)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=configs.epochs, T_mult=1,
                                                                    eta_min=configs.min_lr)
    elif configs.lr_scheduler == 'MultLR':
        print(configs.lr_scheduler)
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=configs.lr_step, gamma=configs.lr_gamma)
    else:
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=configs.step_size, gamma=configs.lr_gamma)

    # 数据加载器
    dataloader = {'train': DataLoader(datasets['train'], batch_size=configs.batch_size, shuffle=True,
                                      num_workers=configs.num_workers,collate_fn=lpcdet_dataset_collate, pin_memory=True),
                  'val': DataLoader(datasets['val'], batch_size=configs.batch_size, shuffle=False,
                                    num_workers=configs.num_workers, collate_fn=lpcdet_dataset_collate, pin_memory=True)}

    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}  # 训练集/验证集的大小
    print('training dataset loaded with length: {}'.format(dataset_sizes['train']))
    print('validation dataset loaded with length: {}'.format(dataset_sizes['val']))

    # 保存训练日志和权重
    save_dir = os.path.join(configs.save_path, time.strftime("%Y%m%d-%H_%M", time.localtime()))
    os.makedirs(save_dir)
    save_weight_dir = save_dir + '/weight'
    save_info_dir = save_dir + '/info'
    os.makedirs(save_weight_dir)
    os.makedirs(save_info_dir)
    train_log_file = save_info_dir + '/{}.txt'.format(time.strftime("%Y%m%d-%H_%M", time.localtime()))
    with open(os.path.join(save_info_dir, 'configs_info.txt'), 'a', encoding='utf8') as f:
        f.write("---------%s training configuration information:---------\n" % model_name)
        f.write("train logs: {}\n".format(save_dir))
        for conf in configs:
            f.write("{}: {}\n".format(conf, configs[conf]))

    start_time = time.time()
    loss_list, val_loss_list, val_acc_list, lr_list = [], [], [], []
    best_acc = 0.1
    epoch_acc = 0.0

    for epoch in range(configs.epochs):
        # flag = True
        print('-' * 22)
        print('Epoch {}/{}'.format(epoch, configs.epochs - 1))
        # 打印并保存当前epoch 的学习率
        current_model_lr = optimizer.state_dict()['param_groups'][0]['lr']
        lr_list.append(current_model_lr)
        print('current {} optimizer of lr:{}'.format(model_name, current_model_lr))
        with open(train_log_file, 'a') as f:
            f.write('current {} optimizer of lr:{}\n'.format(model_name, current_model_lr))

        # 每一个epoch 分为train 和validation 两个阶段
        for phase in ['train', 'val']:
            # train model
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_acc = 0
            # Iterative image data
            for data in tqdm(dataloader[phase], desc=phase + ' Data Processing Progress'):
                bs_images, bs_box_hms, bs_corner_points, bs_corner_hms, bs_corner_regs, bs_corner_masks, gt = data
                bs_image = bs_images.to(device)
                bs_box_hm = bs_box_hms.to(device)
                bs_corner_point = bs_corner_points.to(device)
                bs_corner_hm = bs_corner_hms.to(device)
                bs_corner_reg = bs_corner_regs.to(device)
                bs_corner_reg_mask = bs_corner_masks.to(device)

                # clear gradient
                optimizer.zero_grad()
                # forward propagation, track history only if train
                with torch.set_grad_enabled(phase == 'train'):
                    box_heatmap, corner_heatmap, corner_offset, corner_point = model(bs_image)
                    # 检测网络的损失计算
                    box_hm_loss = configs.scale[0] * focal_loss(box_heatmap, bs_box_hm)
                    corner_hm_loss = configs.scale[1] * focal_loss(corner_heatmap, bs_corner_hm)
                    corner_offset_loss = configs.scale[2] * reg_l1_loss_c(corner_offset, bs_corner_reg, bs_corner_reg_mask)
                    corner_points_loss = configs.scale[3] * reg_l1_loss_c(corner_point, bs_corner_point, bs_corner_reg_mask)  # 0.1
                    L_slack = configs.scale[4] * slack_loss(corner_heatmap, corner_offset, box_heatmap, corner_point)
                    loss = box_hm_loss + corner_hm_loss + corner_offset_loss + corner_points_loss + L_slack
#                     loss = box_hm_loss + corner_hm_loss + corner_offset_loss + corner_points_loss

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    elif epoch > configs.calc_epoch:  # 计算指标
                        det_result = FeatureDecoder.decode_corner(corner_heatmap, corner_offset)
                        det_result = DataProcessor.postprocess_corner(det_result, image_shape=configs.image_shape)
                        for bs in range(bs_corner_hms.shape[0]):
                            IoU = DataProcessor.bbox_iou_eval(gt[bs], det_result[bs])
                            if IoU > configs.iou_threshold:
                                running_acc += 1

                # statistics loss
                running_loss += loss.item()

            epoch_loss = running_loss / dataset_sizes[phase]

            # save train logs and adjust learning rate regulator
            if phase == 'train':
                exp_lr_scheduler.step()
                loss_list.append(epoch_loss)
            else:
                val_loss_list.append(epoch_loss)
                epoch_acc = running_acc / dataset_sizes[phase]
                val_acc_list.append(epoch_acc)
                print('val accuracy:', epoch_acc)

            print('{} Loss:{:.8f}'.format(phase, epoch_loss))
            with open(train_log_file, 'a') as f:
                f.write('Epoch {}/{},\tphase:{},\tloss:{:.6f},\tval_acc:{:.5f},\tcurrent_{}_lr:{:.6f}\n'.format(
                    epoch, configs.epochs, phase, epoch_loss, epoch_acc, model_name, current_model_lr))

            # save weight
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                # Save the current best weight
                if len(os.listdir(save_weight_dir)):
                    filename = os.listdir(save_weight_dir)[0]
                    os.remove(os.path.join(save_weight_dir, filename))
                    torch.save(model.state_dict(), os.path.join(save_weight_dir, '{}_epoch{}_acc{:.6f}_loss{:.4f}.pth'.
                                                                format(model_name, epoch, epoch_acc, epoch_loss)))
                else:
                    torch.save(model.state_dict(), os.path.join(save_weight_dir, '{}_epoch{}_acc{:.6f}_loss{:.4f}.pth'.
                                                                format(model_name, epoch, epoch_acc, epoch_loss)))

        # Save the last generation of training
        if epoch == configs.epochs - 1:
            torch.save(model.state_dict(), os.path.join(save_weight_dir, '{}_last_acc{:.6f}_loss{:.4f}.pth'.format(
                model_name, epoch_acc, epoch_loss)))
        # Plot the training loss curve every fixed epoch
        if epoch % configs.draw_epoch == 0:
            # DataVisualizer.save_info_when_training(loss_list, val_loss_list, val_acc_list, lr_list, save_info_dir)
            DataVisualizer.save_info_when_training(loss_list, val_loss_list, val_acc_list, lr_list, save_info_dir)

    time_spend = time.time() - start_time
    with open(train_log_file, 'a') as f:
        f.write('The total time spent training the model is {:.0f}h{:.0f}m{:.0f}.\n\n'.format(time_spend // 3600, time_spend % 3600 // 60, time_spend % 60))
        f.write("\ntrain_loss:")
        f.write(",".join(str(round(_, 5)) for _ in loss_list))
        f.write("\nval_loss:")
        f.write(",".join(str(round(_, 5)) for _ in val_loss_list))
        
    def filter_value(loss, max_loss=0.5):
        for i, l in enumerate(loss):
            if l > max_loss:
                loss[i] = max_loss
        return loss

    loss_list = filter_value(loss_list)
    val_loss_list = filter_value(val_loss_list)
    DataVisualizer.save_info_when_training(loss_list, val_loss_list, val_acc_list, lr_list, save_info_dir)
    print('The total time spent training the model is {:.0f}h{:.0f}m{:.0f}s'.format(time_spend // 3600, time_spend % 3600 // 60, time_spend % 60))


if __name__ == '__main__':
    # 设定随机种子
    UtilTools.set_random_seed(configs.seed)

    print('train model')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    # 定义参数
    print(configs)
    # 读取训练数据集
    dataset = {'train': LPDataset(img_dir=configs.train_dir['train'],
                                  input_shape=(configs.img_H, configs.img_W), num_classes=configs.num_classes, train=True),
               'val': LPDataset(img_dir=configs.train_dir['val'],
                                input_shape=(configs.img_H, configs.img_W), num_classes=configs.num_classes, train=False)}

    # 文一服务器加载器
    # dataset = {'train': LPDataset(img_dir=configs.train_data_path, txt_path=configs.train_txt_path['train'],
    #                               input_shape=(configs.img_H, configs.img_W), num_classes=configs.num_classes,
    #                               train=True),
    #            'val': LPDataset(img_dir=configs.train_data_path, txt_path=configs.train_txt_path['val'],
    #                             input_shape=(configs.img_H, configs.img_W), num_classes=configs.num_classes,
    #                             train=False)}

    # 训练设备
    train_device = torch.device(configs.device if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.get_device_name(train_device))

    # 实例化模型并加载权重
    lpcdet = LPCDet(model_config=configs.model_configs).to(train_device)
    if configs.weight is not None:
        lpcdet.load_state_dict(torch.load(configs.weight, map_location='cpu'), strict=False)

    for i in range(1):
        train(datasets=dataset,
              model=lpcdet,
              device=train_device
              )
        time.sleep(2)
    print('done')
