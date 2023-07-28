#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2022/12/3 10:25
@Message: null
"""
from matplotlib import pyplot as plt

from utils.util import DataVisualizer


def filter_value(loss, max_loss=0.5):
    for i, l in enumerate(loss):
        if l > max_loss:
            loss[i] = max_loss
    return loss

def loss_curve(train_loss0, val_loss0, train_loss1=None, val_loss1=None, train_loss2=None, val_loss2=None):

    line_width = 2
    plt.figure(figsize=(12, 7), dpi=100)
    plt.subplot(1, 2, 1)
    x_train = range(0, len(train_loss0))
    plt.plot(x_train, train_loss0, color='goldenrod', label='DisLoss', linewidth=line_width)
    if train_loss1 is not None:
        plt.plot(x_train, train_loss1, color='red', label='ChainLoss-MD', linewidth=line_width)
    if train_loss2 is not None:
        plt.plot(x_train, train_loss2, color='steelblue', label='ChainLoss-ED', linewidth=line_width)

    plt.xlabel('epochs')
    plt.ylabel('training loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x_train, val_loss0, color='goldenrod', label='DisLoss', linewidth=line_width)
    if val_loss1 is not None:
        plt.plot(x_train, val_loss1, color='red', label='ChainLoss-MD', linewidth=line_width)
    if val_loss2 is not None:
        plt.plot(x_train, val_loss2, color='steelblue', label='ChainLoss-ED', linewidth=line_width)
    plt.xlabel('epochs')
    plt.ylabel('val loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print()
    #
    print('success')
