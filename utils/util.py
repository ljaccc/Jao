#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2022/7/31 12:45
@Message: null
"""
import os
import random
import zipfile

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
import shapely
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]

ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


class DataVisualizer(object):
    """
    数据可视化的类：
        各种数据类型可视化为图像、
        图像中检测框的可视化、
        将数据可视化后以图片形式保存
    """
    def __init__(self):
        super(DataVisualizer, self).__init__()

    @staticmethod
    def draw_boxes_when_calculating_IoU(gt, det, iou=0, path='data/one'):
        import os
        file_name = os.listdir(path)[0]
        file_path = os.path.join(path, file_name)
        save_path = 'output1/' + file_name.split('-')[0] + '_' + str(round(iou, 2)) + '.jpg'
        im = cv2.imread(file_path)
        cv2.polylines(im, np.expand_dims(gt, axis=0), 1, color=(0, 255, 0), thickness=2)
        cv2.polylines(im, np.expand_dims(det, axis=0), 1, (0, 0, 255), thickness=2)
        cv2.imwrite(save_path, im)

    @staticmethod
    def draw_box(img, points, color=(255, 0, 0), line_width=2, save_path='img_with_box.jpg', save=True):
        points = np.array(points, dtype=np.int32)
        cv2.polylines(img, points, 1, color=color, thickness=line_width)
        if save:
            cv2.imwrite(save_path, img)

    @staticmethod
    def save_loss_curve_when_training(loss_list, val_loss_list, save_path='', save=True):
        x_train = range(0, len(loss_list))
        y_train = loss_list
        plt.figure()
        plt.plot(x_train, y_train, 'b.-', label='train')
        plt.title('train/val loss vs epochs item')
        plt.ylabel('train/val loss')
        x_val = range(0, len(val_loss_list))
        y_val = val_loss_list
        plt.plot(x_val, y_val, 'r.-', label='val')

        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(save_path + '/train_info.png')
        else:
            plt.show()

    @staticmethod
    def save_info_when_training(train_loss, val_loss, val_acc, lr_list, save_path='', save=True):
        X = range(0, len(train_loss))
        y_train = train_loss
        plt.figure(figsize=(12, 7), dpi=200)
        plt.subplot(1, 3, 1)
        plt.plot(X, lr_list, 'y.-', label='lr')
        plt.title('lr vs epochs item')
        plt.ylabel('lr')

        plt.subplot(1, 3, 2)
        plt.plot(X, y_train, 'b.-', label='train')
        plt.title('train/val loss vs epochs item')
        plt.ylabel('train/val loss')
        y_val = val_loss
        plt.plot(X, y_val, 'r.-', label='val')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(X, val_acc, 'g.-', label='val')
        plt.title('val accuracy vs epochs item')
        plt.ylabel('val accuracy')

        plt.tight_layout()
        if save:
            plt.savefig(save_path + '/train_info.png')
        else:
            plt.show()

    @staticmethod
    def draw_box_when_detect(image, top_label, top_conf, top_boxes, crop=False, count=False):
        import os
        from PIL import ImageFont
        # 种类
        num_classes = 1
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='utils/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // 512, 1)
        # ---------------------------------------------------------#
        #   是否进行目标的裁剪
        # ---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            # predicted_class = self.class_names[int(c)]
            predicted_class = 'LP'
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            # print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle((left + i, top + i, right - i, bottom - i), outline=(0, 205, 0))
            draw.rectangle((tuple(text_origin), tuple(text_origin + label_size)))
            draw.text(tuple(text_origin), str(label, 'UTF-8'), fill=(255, 0, 0), font=font)
            del draw
        return image

    @staticmethod
    def show_boxes_on_image(image, points, color='green'):
        if len(points) == 1:
            points = points[0]
        plt.imshow(image)
        x, y = [], []
        for i_p in range(len(points) // 2):
            x.append(points[i_p * 2])
            y.append(points[i_p * 2 + 1])
        if len(points) // 2 == 6:
            plt.scatter(x[2:], y[2:], marker='x', color=color)
        else:
            plt.scatter(x, y, marker='x', color=color)
        plt.show()

    @staticmethod
    def show_feature_map(feature, chanel=41):
        # 将features以图片形式输出  (1, 128, 128, 128)
        feature_temp = feature[:, [chanel, chanel + 1, chanel + 2], :, :].clone()  # (1, 128, 128)
        # feature_temp = features.clone()  # (128, 128, 128)
        feature_temp = torch.squeeze(feature_temp)
        inp = feature_temp.detach().numpy().transpose((1, 2, 0))
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        plt.show()


class DataProcessor(object):
    def __init__(self):
        super(DataProcessor, self).__init__()

    @staticmethod
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

    @staticmethod
    def resize_image(image, size, undistorted=True):
        iw, ih = image.size
        w, h = size
        if undistorted:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        else:
            new_image = image.resize((w, h), Image.BICUBIC)

        return new_image

    @staticmethod
    def decode_label(img_name):
        filename, _ = os.path.splitext(img_name)
        plist = filename.split('-')
        bbox = plist[2].split('_')
        box = [[int(pt.split('&')[0]), int(pt.split('&')[1])] for pt in bbox]
        box_corner = sum(box, [])
        pts = plist[3].split('_')
        pts = np.array(pts)[[2, 3, 0, 1]]
        for pt in pts:
            box_corner.append(int(pt.split('&')[0]))
            box_corner.append(int(pt.split('&')[1]))

        box_corner.append(0)
        box_corner = np.array([box_corner])
        return box_corner

    @staticmethod
    # def decode_label_jxlpd(img_name):
    #     filename, _ = os.path.splitext(img_name)
    #     plist = filename.split('_')
    #     bbox = plist[1:]
    #     box = [[int(pt.split('&')[0]), int(pt.split('&')[1])] for pt in bbox]
    #     box_corner = sum(box, [])
    #     """
    #     box_corner.append(0)
    #     box_corner = np.array([box_corner])
    #     """
    #
    #     #对垂直旋转的图片标注换位置
    #     new_corner = []
    #     for x in box_corner[6:8]:
    #         new_corner.append(x)
    #
    #     for x in box_corner[10:12]:
    #         new_corner.append(x)
    #
    #     for x in box_corner[6:12]:
    #         new_corner.append(x)
    #
    #     for x in box_corner[4:6]:
    #         new_corner.append(x)
    #
    #     new_corner.append(0)
    #     new_corner = np.array([new_corner])
    #
    #     return new_corner
    def decode_label_jxlpd(img_name):
        filename, _ = os.path.splitext(img_name)
        plist = filename.split('_')
        bbox = plist[1:]
        box = [[int(pt.split('&')[0]), int(pt.split('&')[1])] for pt in bbox]
        box_corner = sum(box, [])
        box_corner.append(0)
        box_corner = np.array([box_corner])
        return box_corner

    @staticmethod
    def decode_lp(img_name):
        """
        输入车牌图片的名称，将其解码成车牌字符返回
            img_name: 车牌图片名称
        """

        label_list = img_name.split('-')
        lp_list = label_list[4].split('_')
        lp = ''
        for i, l in enumerate(lp_list):
            if i == 0:
                lp += provinces[int(l)]
            else:
                lp += ads[int(l)]

        return lp

    @staticmethod
    def encode_lp(lp_label):
        """
        输入车牌字符，将其编码成车牌码
            lp_label: 车牌字符
        """
        lp_encode = ''
        for i, l in enumerate(lp_label):
            if i == 0:
                lp_encode += str(provinces.index(l))
            else:
                lp_encode += str(ads.index(l))
            lp_encode += '_'
        return lp_encode[:-1]

    @staticmethod
    def is_clockwise(points):
        if type(points) is not np.ndarray:
            points = np.array(points)
        if points.ndim == 1:
            point = [
                [int(points[0]), int(points[1])],
                [int(points[2]), int(points[3])],
                [int(points[4]), int(points[5])],
                [int(points[6]), int(points[7])]
            ]
        else:
            point = points
        edge = [
            (point[1][0] - point[0][0]) * (point[1][1] + point[0][1]),
            (point[2][0] - point[1][0]) * (point[2][1] + point[1][1]),
            (point[3][0] - point[2][0]) * (point[3][1] + point[2][1]),
            (point[0][0] - point[3][0]) * (point[0][1] + point[3][1])
        ]
        sum_edge = edge[0] + edge[1] + edge[2] + edge[3]
        if sum_edge > 0:
            print('illegal coordinates')
            return False
        return True

    @staticmethod
    def preprocess_input(image):
        image = np.array(image, dtype=np.float32)[:, :, ::-1]
        mean = [0.40789655, 0.44719303, 0.47026116]
        std = [0.2886383, 0.27408165, 0.27809834]
        return (image / 255. - mean) / std

    @staticmethod
    def postprocess_corner(prediction, image_shape=None, input_shape=None, undistorted=True):
        image_shape = np.array(image_shape)
        if image_shape is None:
            image_shape = np.array([1160, 720])
        if input_shape is None:
            input_shape = np.array([512, 512])

        image_shape[0], image_shape[1] = image_shape[1], image_shape[0]
        output = [None for _ in range(len(prediction))]
        # 预测只用一张图片，只会进行一次
        for i, image_pred in enumerate(prediction):
            output[i] = prediction[i]
            if undistorted:
                # -----------------------------------------------------------------#
                #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
                #   new_shape指的是宽高缩放情况
                # -----------------------------------------------------------------#
                new_shape = np.round(image_shape * np.min(input_shape / image_shape))
                offset = (input_shape - new_shape) / 2. / input_shape
                scale = input_shape / new_shape
                output[i][:, ] = (output[i][:, ] - offset) * scale

            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                output[i][:, 0] = np.ceil(output[i][:, 0] * image_shape[0])
                output[i][:, 1] = np.trunc(output[i][:, 1] * image_shape[1])

        return output

    @staticmethod
    def bbox_iou_eval(box1, box2):
        box1 = np.array(box1, dtype=np.int32).reshape(4, 2)
        poly1 = Polygon(box1).convex_hull
        box2 = np.array(box2, dtype=np.int32).reshape(4, 2)
        poly2 = Polygon(box2).convex_hull
        lp_iou = 0
        try:
            inter_area = poly1.intersection(poly2).area  # 相交面积
            lp_iou = float(inter_area) / (poly1.area + poly2.area - inter_area)
            # draw_box(box1, box2, lp_iou)
        except shapely.geos.TopologicalError:
            # print('shapely.geos.TopologicalError occured, iou set to 0')
            print()
        return lp_iou


class FileProcessor(object):
    def __init__(self):
        super(FileProcessor, self).__init__()

    @staticmethod
    def lpcdnet_result(num, gt_points, det_points, gt_path, det_path):
        gt_name = 'gt_img_' + str(num) + '.txt'
        det_name = 'res_img_' + str(num) + '.txt'
        if not os.path.exists(gt_path):
            os.makedirs(gt_path)
        if not os.path.exists(det_path):
            os.makedirs(det_path)

        gt_file_name = os.path.join(gt_path, gt_name)
        det_file_name = os.path.join(det_path, det_name)
        with open(gt_file_name, 'w') as f:
            f.write(gt_points)

        with open(det_file_name, 'w') as f:
            f.write(det_points)

    @staticmethod
    def lpcdnet_result2zip(file_dir, zip_path):
        for file in tqdm(os.listdir(file_dir)):
            file_path = os.path.join(file_dir, file)

            zip_file = zipfile.ZipFile(zip_path, 'a', zipfile.ZIP_DEFLATED)
            zip_file.write(file_path, file)
            zip_file.close()


class GaussianHeatmap(object):
    """
    生成热图的类：
        通过高斯函数函数将坐标映射为热图
    """
    def __init__(self):
        super(GaussianHeatmap, self).__init__()

    @staticmethod
    def gaussian2D(shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    @staticmethod
    def gaussian_radius(det_size, min_overlap=0.7):
        height, width = det_size

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return min(r1, r2, r3)

    @staticmethod
    def draw_gaussian(heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = GaussianHeatmap.gaussian2D((diameter, diameter), sigma=diameter / 6)
        x, y = int(center[0]), int(center[1])
        height, width = heatmap.shape[0:2]
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap


class FeatureDecoder(object):
    """
    特征解码器：
        将模型中的特征进行解码
    """

    def __init__(self):
        super(FeatureDecoder, self).__init__()

    @staticmethod
    def pool_nms(heatmap, kernel=3):
        pad = (kernel - 1) // 2
        hmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heatmap).float()
        return heatmap * keep

    @staticmethod
    def decode_corner(pre_hms, pre_offsets):
        # 热图的非极大值抑制，利用3×3 的卷积对热图进行最大值筛选，找出区域内得分最大的特征
        pre_hms = FeatureDecoder.pool_nms(pre_hms)  # (1,4,128,128)
        # 这里应该是 bs, C, H, W，不太确定
        batch, c, output_h, output_w = pre_hms.shape
        detects = []
        for b in range(batch):
            # heatmap   (128,128,num_classes)   corner 的热图
            corner_heatmap = pre_hms[b].permute(1, 2, 0).view([-1, c])  # (16384,4)
            pred_offsets = pre_offsets[b].permute(1, 2, 0).view([-1, 8])  # (16384,8)
            # corner 的4个点坐标，即每个corner_heatmap 上置信度最大的预测
            xy_list = []
            for corner_heatmap_i in range(4):
                corner_conf, corner_pred = torch.max(corner_heatmap[:, corner_heatmap_i], dim=-1)
                if pre_offsets is not None:
                    x_offset = pred_offsets[corner_pred, corner_heatmap_i * 2]
                    y_offset = pred_offsets[corner_pred, corner_heatmap_i * 2 + 1]
                    # 每个角点中心+对应的偏移量
                    x = corner_pred % 128 + x_offset
                    y = corner_pred / 128 + y_offset
                else:
                    x = corner_pred % 128
                    y = corner_pred / 128
                xy_list.append([x, y])
            # 将其合并
            corners = torch.Tensor(xy_list)  # (4, 2)
            corners[:, [0]] /= output_w
            corners[:, [1]] /= output_h
            detects.append(corners)

        return detects

    @staticmethod
    def decode_corner_by_center(pred_bhm, pred_points):
        pred_bhm = FeatureDecoder.pool_nms(pred_bhm)
        batch, c, h, w = pred_bhm.shape
        detects = []
        for b in range(batch):
            # box2corner
            box_heatmap = pred_bhm[b].permute(1, 2, 0).view([-1, c])  # (16384,1)
            corner_point = pred_points[b].permute(1, 2, 0).view([-1, 8])  # (16384, 8)
            center_conf, center_pred = torch.max(box_heatmap[..., 0], dim=-1)
            x_center = center_pred % 128
            y_center = center_pred / 128

            xy_list = []
            for corner_i in range(4):
                x_point = corner_point[center_pred, corner_i * 2] + x_center
                y_point = corner_point[center_pred, corner_i * 2 + 1] + y_center
                xy_list.append([x_point, y_point])

            corners = torch.Tensor(xy_list)  # (4, 2)
            corners[:, [0]] /= w
            corners[:, [1]] /= h
            detects.append(corners)
        return detects


class UtilTools(object):
    """
    工具类
    """

    def __init__(self):
        super(UtilTools, self).__init__()

    @staticmethod
    def set_random_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        print('You have chosen to seed training. This will slow down your training!')