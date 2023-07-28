# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2022/3/18 10:27
@Message:
包含的标注信息提取出来用于YOLOv5训练
具体内容：src_dir为包含车牌图片的源文件夹
功能：将车牌图片和对应的json标注信息提取出来作为图片名称
具体内容：将包含车牌图片和json信息的源文件夹路径赋值给src_dir,
         将含有标注信息的图片存入指定的目标文件夹dist_dir,
         若图片和json没有对应文件，则忽视该文件
"""
import copy
import json
import os
import numpy as np
import cv2

from PIL import Image, ImageDraw

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]


def json2file_name(file_json, file_img, directory_save):
    """
    将json文件中的标注转换为图片文件名
    """
    points = []
    # read json
    with open(file_json, 'r', encoding='utf-8') as f:
        b = f.read()
        data = json.loads(b)
    # get message from json
    for coordict in data['shapes']:
        coordlist = coordict['points']
        for point in coordlist:
            for i in range(len(point)):
                point[i] = int(point[i])
            points.append(point)

        label_list = coordict['label']

    # 计算以哪个对角点作为框
    max = (points[2][0] - points[0][0]) ** 2 + (points[2][1] - points[0][1]) ** 2
    if max > ((points[1][0] - points[3][0]) ** 2 + (points[1][1] - points[3][1]) ** 2):
        points.insert(0, [points[0][0], points[0][1] - 5])
        points.insert(1, [points[3][0] + 5, points[3][1]])
    else:
        temp = [points[1][0] + 5, points[3][1]]

        points.insert(0, [points[3][0], points[1][1] - 5])
        points.insert(1, temp)

    img = cv2.imdecode(np.fromfile(file_img, dtype=np.uint8), 1)  # 可读取中文路径图片

    img_name = str(provinces.index(label_list[0])) + '&' + label_list[1:]
    for point in points:
        img_name = img_name + '_' + str(point[0]) + "&" + str(point[1])

    cv2.imwrite(directory_save + '/' + img_name + '.jpg', img)


def read_directory(directory_name, directory_save):
    """读取文件夹中的文件，并将json标签转换为图片名称"""
    for item in os.listdir(directory_name):
        if item[-4:] == 'json':
            file_json = directory_name + '/' + item
            file_img = directory_name + '/' + item[:-4] + 'jpg'
            try:
                json2file_name(file_json, file_img, directory_save)
            except:
                continue
    print("Done")



def resize_image_with_label(directory_name, dist_dir, target_size=None):
    def resize_image_with_boxes(image, box_corner, size=None, undistorted=True):
        # ---------------------------------------------------------
        # 类型为 img=Image.open(path)，boxes:Tensor，size:int
        # 功能为：将图像长和宽缩放到指定值size，并且相应调整boxes
        # ---------------------------------------------------------
        if size is None:
            size = [512, 512]
        iw, ih = image.size  # input size
        tw, th = size  # target size
        if undistorted:
            scale = min(tw / iw, th / ih)
            cw, ch = int(iw * scale), int(ih * scale)  # current size
            dw, dh = int((tw - cw) / 2), int((th - ch) / 2)  # size to be filled
            image_fill = image.resize((cw, ch), Image.BICUBIC)
            image_resize = Image.new('RGB', (tw, th), (0, 0, 0))
            image_resize.paste(image_fill, (dw, dh))
            box_corner[:, [0, 2, 4, 6, 8, 10]] = box_corner[:, [0, 2, 4, 6, 8, 10]] * cw / iw + dw
            box_corner[:, [1, 3, 5, 7, 9, 11]] = box_corner[:, [1, 3, 5, 7, 9, 11]] * ch / ih + dh
        else:
            scale_w, scale_h = tw / iw, th / ih
            image_resize = image.resize((tw, th), Image.BICUBIC)
            box_corner[:, [0, 2, 4, 6, 8, 10]] = box_corner[:, [0, 2, 4, 6, 8, 10]] * scale_w
            box_corner[:, [1, 3, 5, 7, 9, 11]] = box_corner[:, [1, 3, 5, 7, 9, 11]] * scale_h

        # 判断异常坐标，并进行修正
        # 当左上角和左下x坐标越界，设为0
        box_corner[:, [0, 1, 4, 5, 10]][box_corner[:, [0, 1, 4, 5, 10]] < 0] = 0
        # 当右下角和右上x坐标、左下y和右上y坐标越界，则分别为image 的w、h
        box_corner[:, [2, 6, 8]][box_corner[:, [2, 6, 8]] > tw] = tw
        box_corner[:, [3, 7, 9, 11]][box_corner[:, [3, 7, 9, 11]] > th] = th
        return image_resize, box_corner
    def draw_box(img, points, color=(255, 0, 0), line_width=2, save_path='img_with_box.jpg', save=True):
        points = np.array(points, dtype=np.int32)
        cv2.polylines(img, points, 1, color=color, thickness=line_width)
        if save:
            cv2.imwrite(save_path, img)

    if target_size is None:
        target_size = [520, 304]
    for item in os.listdir(directory_name):
        img_name, suffix = os.path.splitext(item)

        img_path = os.path.join(directory_name, item)
        img = Image.open(img_path)

        img_label = img_name.split('_')
        labels = []
        for lb in img_label[1:]:
            lbs = lb.split('&')
            labels.append(int(lbs[0]))
            labels.append(int(lbs[1]))

        labels = np.array([labels])

        # 调整前
        labels_show = np.array(labels[:, 4:]).reshape(1,4,2)
        img_cv = copy.deepcopy(img)
        img_cv = cv2.cvtColor(np.asarray(img_cv), cv2.COLOR_RGB2BGR)
        draw_box(img_cv, labels_show)

        new_img, new_labels = resize_image_with_boxes(img, labels, size=target_size, undistorted=False)

        #
        # 调整后
        labels_show = np.array(new_labels[:, 4:]).reshape(1, 4, 2)
        img_cv = cv2.cvtColor(np.asarray(new_img), cv2.COLOR_RGB2BGR)
        draw_box(img_cv, labels_show, save_path='img_with_box2.jpg')
        print()

        new_file_name = img_label[0]
        for index, l in enumerate(new_labels[0]):
            temp = ''
            if index % 2 == 0:
                temp += '_' + str(l)
            else:
                temp +=  ('&' + str(l))

            new_file_name += temp

        new_img_path = os.path.join(dist_dir, new_file_name+suffix)
        new_img.save(new_img_path)

    print("done")

if __name__ == '__main__':
    # src_dir = 'data/todo'  # # 待转换的文件夹
    # dist_dir = 'data/output'  # 目标文件夹，即转换后存放的文件夹
    # # exit("请先修改为待转换的文件路径，再将该语句注释！")
    # read_directory(src_dir, dist_dir)
    directory_name = 'data/dev/dst'
    dist_dir = "data/dev/src"
    resize_image_with_label(directory_name, dist_dir, [520, 304])

    # labels = np.array([labels])
    #
    # # 调整前
    # labels_show = np.array(labels[:, 4:]).reshape(1,4,2)
    # img_cv = copy.deepcopy(img)
    # img_cv = cv2.cvtColor(np.asarray(img_cv), cv2.COLOR_RGB2BGR)
    # draw_box(img_cv, labels_show)
    #
    # new_img, new_labels = resize_image_with_boxes(img, labels, size=[520, 304], undistorted=False)
    #
    # # 调整后
    # labels_show = np.array(new_labels[:, 4:]).reshape(1, 4, 2)
    # img_cv = cv2.cvtColor(np.asarray(new_img), cv2.COLOR_RGB2BGR)
    # draw_box(img_cv, labels_show, save_path='img_with_box2.jpg')
    # print()
