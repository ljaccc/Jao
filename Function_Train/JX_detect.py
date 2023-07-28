import argparse
import codecs
import csv
import json
import os
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from models.LPCDet import LPCDet
from utils.util import DataProcessor, FeatureDecoder, DataVisualizer


def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存csv文件成功，处理结束")


def run(opts, model, device):
    # 加载模型权重
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(opts.weight))
    save_path = opts.save_path
    count = len(os.listdir(r'F:\lp_data\test_acc'))
    print(count)
    res_all = []
    for file_name in tqdm(os.listdir(r'F:\lp_data\test_acc')):
        file_path = os.path.join(r'F:\lp_data\test_acc', file_name)
        # 读取图片
        image = Image.open(file_path)
        image_np = np.array(image)
        image_shape = image_np.shape
        img_name = file_path.split('\\')[-1]
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
        res = np.array(results[0], dtype=np.int32).tolist()
        res_all.append(res)

        #print(res)
        # 展示检测结果，默认False
        if opts.show:
            img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            DataVisualizer.draw_box(img, results, color=(0, 255, 255), save=False)
            DataVisualizer.draw_box(img, results, color=(0, 255, 0), save_path=save_path+'/'+img_name)

    return res_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default=r'F:\6Model_deployment\LPCDet_L\run\20230507-19_28\weight\LPCDet_best.pth', help='weight of LPCDet_L')
    parser.add_argument('--threshold', default=0.9, help='IoU')
    parser.add_argument('--image_dir', type=str, default='data/onlyone', help='input image')
    parser.add_argument('--save_path', type=str, default='output7.27', help='input image')
    parser.add_argument('--input_size', default=[512, 512], help='input image size')
    parser.add_argument('--undistorted', default=True, help='input image size')
    parser.add_argument('--show', default=False, help='are you show?')
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
    json_file_path = r'/kk2.json'
    json_file = open(json_file_path, mode='w')
    save_json_content = []
    for r in res:
        save_json_content.append(r)

    json.dump(save_json_content, json_file, indent=4)




