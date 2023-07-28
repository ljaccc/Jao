#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2023/05/01 00:07
@Message: null
"""

import os
import time
import zipfile
from tqdm import tqdm
from eval2file import gt_path, det_path


def lpcdnet_result2zip(file_dir, zip_path):
    for file in tqdm(os.listdir(file_dir)):
        file_path = os.path.join(file_dir, file)

        zip_file = zipfile.ZipFile(zip_path, 'a', zipfile.ZIP_DEFLATED)
        zip_file.write(file_path, file)
        zip_file.close()


if __name__ == '__main__':
    print("eval2zip")
    start_time = time.time()

    gt_path = gt_path     # 包含gt的文件夹路径
    det_path = det_path   # 包含检测结果的文件夹路径
    dir_path = 'zip_file'
    for file_name in tqdm(os.listdir(dir_path)):
        file_path = dir_path + '/' + file_name
        zip_path = file_path + '.zip'
        print(file_path)
        print(zip_path)
        assert not os.path.exists(
            zip_path), 'Error: the file path "%s" is exists, append write is unnecessary' % zip_path
        lpcdnet_result2zip(file_path, zip_path)

    time_spend = time.time() - start_time
    print('Eval complete in {:.0f}m {:.0f}s'.format(time_spend // 60, time_spend % 60))


