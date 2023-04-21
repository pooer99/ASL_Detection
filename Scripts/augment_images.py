import albumentations as A
import cv2
import os
from matplotlib import pyplot as plt
import random


'''用于对原有的数据集进行数据增强'''

'''文件根目录'''
dataset_path = r'F:\augment_image_test\input'
images_path = os.path.join(dataset_path, 'images')
labels_path = os.path.join(dataset_path, 'labels')

'''获取图片'''
# 获取图片
file_name = 'A1.png'  # 图片文件名
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
def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=4, **kwargs):
    print('变换后：')
    print(bbox)

    bbox = xyxy_to_xywh(size,bbox)
    print('xyxy转xywh后：')
    print(bbox)

    x_min, y_min, w, h = bbox
    #x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    ax = plt.gca()
    # 默认框的颜色是黑色，第一个参数是左上角的点坐标
    # 第二个参数是宽，第三个参数是长
    ax.add_patch(plt.Rectangle((x_min, y_min), w, h, color="blue", fill=False, linewidth=thickness))
    plt.imshow(img)
    plt.waitforbuttonpress()
    return img

def xyxy_to_xywh(size, bbox):
    """
    将bbox的左上角点、右下角点坐标的格式，转换为bbox中心点 + bbox的w,h的格式，并进行归一化
    size: [weight, height]
    bbox: [Xmin, Ymin, Xmax, Ymax]
    即：xyxy（左上右下） ——> xywh（中心宽高）
    xyxy（左上右下）:左上角的xy坐标和右下角的xy坐标
    xywh（中心宽高）:边界框中心点的xy坐标和图片的宽度和高度
    """
    dw = size[0]
    dh = size[1]
    x = (bbox[0] + bbox[2]) / 2.0
    y = (bbox[1] + bbox[3]) / 2.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = x * dw
    y = y * dh
    w = w * dw
    h = h * dh
    return (x, y, w, h)

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
result = visualize_bbox(transformed_image,transformed_bboxes[0])

#visualize(result)


