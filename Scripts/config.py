import os

#----------用于存储定义的参数----------

#-----训练参数-----

# 图片路径
IMAGE_PATH = os.path.join('../HP_Date/Image')

# 手势标签
labels = ['a', 'a lot', 'abdomen', 'able']

# 每种手势图片数量
number_images = 8

# 获取图片目录下的文件名列表
def getFileName(num):
    file_path = os.path.join(IMAGE_PATH, labels[num])
    return os.listdir(file_path)
