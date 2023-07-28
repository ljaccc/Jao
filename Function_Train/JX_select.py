import os
import cv2
import numpy as np
from tqdm import tqdm
count = 0
for file in tqdm(os.listdir(r'F:\lp_data\20230515')):
       for file_name in tqdm(os.listdir(os.path.join(r'F:\lp_data\20230515',file))):
            file_path = os.path.join(r'F:\lp_data\20230515', file , file_name)
            image = cv2.imread(file_path)
            image_np = np.array(image)
            image_shape = image_np.shape
            if image_shape == (400,300,3):
                load = os.path.join(r'F:\lp_data\select_300x400',file_name)
                try:
                    cv2.imwrite(load, image)
                except:
                    pass

            else:
                load = os.path.join(r'F:\lp_data\select_520x304',file_name)
                try:
                 cv2.imwrite(load, image)
                except:
                    pass
