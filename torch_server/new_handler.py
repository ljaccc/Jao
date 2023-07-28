# custom handler file

# model_handler.py

"""
ModelHandler defines a custom model handler.
"""
import torch
import io
from PIL import Image
from torch import nn
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler
import numpy as np
import os
import cv2
class ModelHandler(BaseHandler,object):

    """
    A custom model handler implementation.
    """

    def __init__(self, **kwargs):
        self.image = None
        self.image_shape = None
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None
        self.image_shape = None
        self.img_preprocess = transforms.Compose([

            # 裁剪
            #transforms.Resize((112, 112)),
            # PIL图像转为tensor，归一化到[0,1]：Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            transforms.ToTensor(),
            # 规范化至 [-1,1]
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

        ])



       
    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """

        #  load the model, refer 'custom handler class' above for details
        #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        self.model = torch.jit.load(model_pt_path)

        self.initialized = True



    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """

        for row in data:######输出为PIL image 格式
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            self.image = row.get("data") or row.get("body")
            ##print(self.image)
            ##print(type(self.image))
            self.image = Image.open(io.BytesIO(self.image))
            #self.image2 = np.array(bytearray(self.image), dtype="uint8")
        image_np = np.array(self.image)
        self.image_shape = image_np.shape
        print(self.image_shape)
        print("=============111================")
        data = self.resize_image(self.image, [512, 512], undistorted=True)
        # 对图片进行归一化和变换通道 (C, W, H)kk
        data = np.expand_dims(np.transpose(self.preprocess_input(np.array(data, dtype=np.float32)), (2, 0, 1)),0)  # (1,3,512,512)
        input = torch.from_numpy(data).type(torch.FloatTensor).to(self.device)
        return input


    def inference(self, input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output7.27 in NDArray
        """
        # Do some inference call to engine here and return output7.27
        box_heatmap, corner_heatmap, corner_offset, corner_point = self.model.forward(input)

        # -----------------------------------------------------------#
        #   利用预测结果进行解码
        # -----------------------------------------------------------#

        return corner_heatmap, corner_offset

    def postprocess(self, corner_heatmap, corner_offset):
        """
        Return inference result.
        :param inference_output: list of inference output7.27
        :return: list of predict results
        """
        outputs = self.decode_corner(corner_heatmap, corner_offset)
        results = self.postprocess_corner(outputs, image_shape=[self.image_shape[0], self.image_shape[1]],undistorted=True)
        res = np.array(results[0], dtype=np.int32).tolist()
        return  [res]
    def handle(self,data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output7.27
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output7.27
        """
        images  = self.preprocess(data)
        corner_heatmap, corner_offset = self.inference(images)
        return self.postprocess(corner_heatmap, corner_offset)

    def preprocess_input(self,image):
        image = np.array(image, dtype=np.float32)[:, :, ::-1]
        mean = [0.40789655, 0.44719303, 0.47026116]
        std = [0.2886383, 0.27408165, 0.27809834]
        return (image / 255. - mean) / std

    def resize_image(self , image, size, undistorted=True):
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

    def decode_corner(self,pre_hms, pre_offsets):
        # 热图的非极大值抑制，利用3×3 的卷积对热图进行最大值筛选，找出区域内得分最大的特征
        pre_hms = self.pool_nms(pre_hms)  # (1,4,128,128)
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

    def pool_nms(self , heatmap, kernel=3):
        pad = (kernel - 1) // 2
        hmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heatmap).float()
        return heatmap * keep

    def postprocess_corner(self , prediction, image_shape=None, input_shape=None, undistorted=True):
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
