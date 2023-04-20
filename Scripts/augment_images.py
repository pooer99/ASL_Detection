import cv2
from matplotlib import pyplot as plt
import albumentations as A
import random


'''用于对原有的数据集进行数据增强'''

'''获取图片并转换为RGB格式'''
img = cv2.imread('F://augment_image_test//G0.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

'''定义可视化函数'''
def visualize(image):
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(image)
    plt.show()

'''定义扩充管道'''
transform = A.Compose([
    A.CLAHE(),
    A.RandomRotate90(),
    A.Transpose(),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
    A.Blur(blur_limit=3),
    A.OpticalDistortion(),
    A.GridDistortion(),
    A.HueSaturationValue(),
])
random.seed(42)

'''开始数据增强'''
transformed_image = transform(image=img)['image']

'''显示图片'''
visualize(transformed_image)


