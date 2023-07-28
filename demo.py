#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2022/7/17 20:05
@Message: null
"""
import math
import os
import time
import zipfile

import cv2
import numpy as np
from matplotlib import pyplot as plt

from torch import nn
import torch

from utils.util import DataProcessor





if __name__ == '__main__':
    # 文件夹路径
    folder_path = r'F:\6Model_deployment\LPCDet_L\data\outs'
    image_shape = [520,304]
    image_shape_post = [1920,1140]
    for filename in os.listdir(folder_path):
        file_num = filename.split('_')
        for i in range(1,5):
            corner = file_num[i].split('&')
            # print(corner)
    # print(configs)

    # 车牌解码
    # label = '01-86_91-298&341_449&414-458&394_308&410_304&357_454&341-0_0_14_28_24_26_29-124-24'
    # lp = decode_lp(label)
    # print(lp)

    # 车牌编码
    # lp_label = '皖A8888X'
    # lp_encode = DataProcessor.encode_lp(lp_label)
    # print(lp_encode)

    # img_name = 'base@0108620689655-91_81-130&434_310&494-321&499_132&499_119&433_308&433-0_0_8_27_25_24_4-115-19.jpg'
    # tres, last = img_name.split('.')
    # print(tres)

    # dir_path = '../dataset/jxlpd_trains'
    #
    # error_count = 0
    # # for i, file_name in enumerate(os.listdir(dir_path)):
    # #     file_check = file_name.split('_')
    # #     if len(file_check) != 7:
    # #         error_count += 1
    # #         print(file_name)
    #
    # print(error_count)








    # x = torch.tensor([[[1, 3, 2], [1, 2, 2]]])
    # c2 = torch.ones(1, 64, 256, 256)
    # c3 = torch.ones(1, 64, 256, 256) * 3
    # print(c2.shape)

    # def demo_draw_loss_function_curve(x_max=30, beta=6):
    #     # 一次性函数：绘制定位约束损失函数的曲线
    #     x_e_true = [round(num, 1) for num in np.arange(0, beta, 0.1)]
    #     x_e_false = [round(num, 1) for num in np.arange(beta, 50, 0.1)]
    #     x_l_true = [round(num, 1) for num in np.arange(beta, 50, 0.1)]
    #     x_l_false = [round(num, 1) for num in np.arange(0, beta, 0.1)]
    #     y_e_true = [cal(i, alph=0.1) for i in x_e_true]
    #     y_e_false = [cal(i, alph=0.1) for i in x_e_false]
    #
    #     y_p = np.arange(0.98, beta, 0.1)
    #     x_piecewise = [beta] * len(y_p)
    #     y_piecewise = [round(num, 1) for num in y_p]
    #     plt.plot(x_e_true, y_e_true, linestyle='-', color='steelblue', linewidth=2, label="$ d_{mht_{i}}$ < $\\beta$")
    #     plt.plot(x_e_false, y_e_false, linestyle='--', color='steelblue', )
    #     plt.plot(x_l_true, x_l_true, linestyle='-', color='sandybrown', linewidth=2, label="$otherwise$")
    #     plt.plot(x_l_false, x_l_false, linestyle='--', color='sandybrown')
    #     plt.plot(x_piecewise, y_piecewise, linestyle='--', color='red')
    #     plt.xlim(0, x_max)
    #     plt.ylim(0, x_max)
    #     plt.legend()
    #     plt.wo_src()


    # demo_draw_loss_function_curve()

    # def demo_draw_lf_derivation(x_max=30, y_max=8, beta=6):
    #     # 绘制定位约束损失函数的求导曲线
    #     x = [round(num, 1) for num in np.arange(0, x_max, 0.1)]
    #     y_l = [1] * len(x)
    #     y_e = [0.1 / beta * math.exp(i / beta) for i in x]
    #
    #     plt.plot(x, y_e, linestyle='-', color='steelblue', linewidth=2,
    #              label="$ y_{1}^{'}=\\frac{0.1}{5}·e^{\\frac{d_{mht_{i}}}{5}}$")
    #     plt.plot(x, y_l, linestyle='-', color='sandybrown', linewidth=2, label="$ y_{2}^{'}=1$")
    #     plt.xlim(0, x_max)
    #     plt.ylim(0, y_max)
    #     plt.legend()
    #     plt.show()
    #
    #
    # # demo_draw_lf_derivation()
    #
    # print()
