import albumentations as A
import cv2
import os
from matplotlib import pyplot as plt
import random

from common_functions import read_bb, write_bb, yolo_to_cv2


'''用于对原有的数据集进行数据增强'''

'''文件根目录'''
dataset_path = r'F:\augment_image_test\input'
images_path = os.path.join(dataset_path, 'images')
labels_path = os.path.join(dataset_path, 'labels')

output_path = r'F:\augment_image_test\out'
'''获取图片'''
# 获取图片
file_name = 'A0.jpg'  # 图片文件名
file_path = os.path.join(images_path, file_name)  # 图片文件路径

img = cv2.imread(file_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 获取图片宽、高
size = img.shape
w = size[1]
h = size[0]

'''获取图片标注'''
# 获取标签
file_label_name = file_name[:-3] + 'txt'  # 根据图片文件名，修改为label的文件名
file_label_path = os.path.join(labels_path, file_label_name)  # label文件路径

# 获取label的边框坐标
bboxes = [[]]
ids = [[]]
label_count = 1  # 标注数量
with open(file_label_path, 'r', encoding='utf-8') as f:
    read = f.readline().rstrip()
    data = read.split(' ')
    # 获取类id
    ids[0] = data[0]
    # 获取边框坐标
    data.remove(data[0])
    bboxes[0] = list(map(float,data))
f.close()

print(ids)
print('变换前：')
print(bboxes)
'''定义可视化函数'''
def visualize(image):
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(image)
    plt.waitforbuttonpress()

BOX_COLOR = (255, 0, 0) # Red
def visualize_bbox(img, bboxes, color=BOX_COLOR, thickness=1, **kwargs):
    print('变换后：')
    print(bboxes[0])

    label_name = 'transfromed_0.txt'
    write_bb(os.path.join(output_path, label_name), bboxes, ids)

    # 将yolo格式边框坐标归一化
    bboxes[0] = yolo_to_cv2(bboxes[0],size)
    x_min, x_max, y_min, y_max = bboxes[0]

    print('xyxy转xywh后：')
    print(bboxes[0])

    # 绘制边框，用于检验
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

    # 保存数据增强后的images和labels
    image_name = 'transfromed_0.jpg'
    cv2.imwrite(os.path.join(output_path,image_name),img)

    cv2.destroyAllWindows()
    return img

'''定义扩充管道'''
random.seed(42)

transform = A.Compose([
    A.HorizontalFlip(p=1)
], bbox_params=A.BboxParams(format='yolo', min_area=1024, label_fields=['class_labels']))

'''开始数据增强'''
transformed = transform(image=img, bboxes=bboxes, class_labels=ids)
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']
transformed_class_labels = transformed['class_labels']

'''显示图片'''
result = visualize_bbox(transformed_image,transformed_bboxes)

visualize(result)


