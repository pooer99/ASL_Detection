import albumentations as A
import cv2
import os
import random

from common_functions import write_bb, read_images, visualize, visualize_bbox


'''定义增强管道,用于对原有的数据集进行数据增强'''
def make_augmentor():

    # 定义bbox参数
    bbox_param = A.BboxParams(format='yolo', label_fields=['class_labels'])
    random.seed(42)

    # 以下是各种图像处理的增强管道的定义：
    # 水平翻转
    augmentors = {'HorizontalFlip': A.Compose([
        A.HorizontalFlip(p=1)
    ], bbox_params=bbox_param)}

    return augmentors


'''获取图片'''
def augment_images(transform):

    # 文件根目录
    dataset_path = r'F:\augment_image_test\input'
    images_path = os.path.join(dataset_path, 'images')
    labels_path = os.path.join(dataset_path, 'labels')

    output_path = r'F:\augment_image_test\out'
    output_images_path = r'F:\augment_image_test\out\images'
    output_labels_path = r'F:\augment_image_test\out\labels'

    # 获取图片
    files_name, files_path = read_images(images_path)  # 图片文件路径

    for i in range(len(files_path)):
        img = cv2.imread(files_path[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 获取图片宽、高
        size = img.shape
        w = size[1]
        h = size[0]

        '''获取图片标注'''
        # 获取标签
        file_label_name = files_name[i][:-3] + 'txt'  # 根据图片文件名，修改为label的文件名
        file_label_path = os.path.join(labels_path, file_label_name)  # label文件路径

        ids = []
        bbox = []
        # 获取label的边框坐标
        with open(file_label_path, 'r', encoding='utf-8') as f:
            read = f.readline().rstrip()
            data = read.split(' ')

            # 获取类id
            ids.append(data[0])
            # 获取边框坐标
            data.remove(data[0])
            bbox.append(list(map(float,data)))

        f.close()

        print(ids)
        print('变换前：')
        print(bbox)

        '''开始数据增强'''
        transformed = transform(image=img, bboxes=bbox, class_labels=ids)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_class_labels = transformed['class_labels']

        print('变换后：')
        print(transformed_bboxes)
        print('-----------------')

        # 保存图片，标签
        transformed_image = cv2.cvtColor(transformed_image,cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(output_images_path, files_name[i]), transformed_image)
        write_bb(os.path.join(output_labels_path, file_label_name), transformed_bboxes, ids)

    cv2.destroyAllWindows()


'''显示图片'''
def show_iamges_with_bboxes(image,bbox):
    result = visualize_bbox(image,bbox)
    visualize(result)

if __name__ == '__main__':
    # 获取增强管道
    augmentors = make_augmentor()

    # 进行图片增强
    augment_images(augmentors['HorizontalFlip'])



