import os, random, shutil

def moveFile(fileDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    rate = 0.7 # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片（如果不想设置比例，可以将rate = 0.2注释掉，直接自定义picknumber数值，如picknumber = 10）
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print(sample)
    for name in sample:
        # shutil.copy(fileDir + name, tarDir + name) # 复制
        shutil.move(fileDir + '\\'+name, tarDir + '\\' + name) # 剪切

    return


if __name__ == '__main__':
    fileDir = r"F:\lp_data\lpd_168x48\CCPD_CRPD_OTHER_ALL"  # 源图片文件夹路径
    tarDir = r'F:\lp_data\lpd_168x48\2023.5.25_train'  # 移动到新的文件夹路径
    moveFile(fileDir)
