import albumentations as A
import cv2
import os
import shutil
import random

from common_functions import write_bb, read_images, random_files, delet_images, random_augment_opt, show_iamges_with_bboxes,random_files_move


'''定义增强管道,用于对原有的数据集进行数据增强'''
def make_augmentor():

    # 定义bbox参数
    bbox_param = A.BboxParams(format='yolo', label_fields=['class_labels'])

    ''' 以下是各种图像处理的增强管道的定义：
        增强方法 opt :
            HorizontalFlip : 水平翻转
            Blur           : 模糊
            RandomScale    : 随机缩放
            ...
    '''
    augmentor = {
        # 以下为像素级：
        # 模糊
        'Blur': A.Compose([
            A.Blur(p=1, always_apply=True)
        ], bbox_params=bbox_param),

        # 随机缩放
        'RandomScale': A.Compose([
            A.RandomScale(scale_limit=0.5, always_apply=True),
        ], bbox_params=bbox_param),

        # 噪声
        'CLAHE': A.Compose([
            A.CLAHE(always_apply=True),
        ], bbox_params=bbox_param),

        # 高斯噪声
        'GaussNoise': A.Compose([
            A.GaussNoise(var_limit=(100.0, 110.0), always_apply=True),
        ], bbox_params=bbox_param),

        # 高斯模糊
        'GlassBlur': A.Compose([
            A.GlassBlur(always_apply=True),
        ], bbox_params=bbox_param),

        # 运动模糊
        'MotionBlur': A.Compose([
            A.MotionBlur(blur_limit=7, always_apply=True),
        ], bbox_params=bbox_param),

        # 随机亮度对比度
        'RandomBrightnessContrast': A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, always_apply=True),
        ], bbox_params=bbox_param),

        # 随机雾
        'RandomFog': A.Compose([
            A.RandomFog(fog_coef_lower=0.4, fog_coef_upper=0.8, always_apply=True),
        ], bbox_params=bbox_param),

        # 随机雨
        'RandomRain': A.Compose([
            A.RandomRain(always_apply=True),
        ], bbox_params=bbox_param),

        # 随机阴影
        'RandomShadow': A.Compose([
            A.RandomShadow(always_apply=True),
        ], bbox_params=bbox_param),

        # 随机光斑
        'RandomSunFlare': A.Compose([
            A.RandomSunFlare(flare_roi=(0.3, 0.3, 0.8, 0.8), src_radius=150, always_apply=True),
        ], bbox_params=bbox_param),

        # 缩放模糊
        'ZoomBlur': A.Compose([
            A.ZoomBlur(always_apply=True),
        ], bbox_params=bbox_param),

        # 色调和饱和度
        'HueSaturationValue': A.Compose([
            A.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=40, val_shift_limit=50,
                                 always_apply=True),
        ], bbox_params=bbox_param),

        # 超像素
        'Superpixels': A.Compose([
            A.Superpixels(p_replace=0.1, n_segments=200, always_apply=True),
        ], bbox_params=bbox_param),

        # 锐化
        'Sharpen': A.Compose([
            A.Sharpen(p=1, always_apply=True),
        ], bbox_params=bbox_param),

        # 离焦
        'Defocus': A.Compose([
            A.Defocus(always_apply=True),
        ], bbox_params=bbox_param),

        # 以下为空间级：
        # 安全旋转
        'SafeRotate': A.Compose([
            A.SafeRotate(limit=60, always_apply=True),
        ], bbox_params=bbox_param),

        # 仿射
        'Affine': A.Compose([
            A.Affine(rotate=[-60, 60], always_apply=True),
        ], bbox_params=bbox_param),

        # 水平翻转
        'HorizontalFlip': A.Compose([
            A.HorizontalFlip(p=1, always_apply=True)
        ], bbox_params=bbox_param),
    }

    return augmentor


'''获取图片'''
def augment_images(augs):

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

        # print(ids)
        # print('变换前：')
        # print(bbox)

        '''开始数据增强'''
        transform, opt = random_augment_opt(augs)  # 随机选择增强方式

        transformed = transform(image=img, bboxes=bbox, class_labels=ids)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_class_labels = transformed['class_labels']

        # print('变换后：')
        # print(transformed_bboxes)
        # print('-----------------')

        # 保存图片，标签
        transformed_image = cv2.cvtColor(transformed_image,cv2.COLOR_BGR2RGB)

        # files_name[i] = opt + '_' + files_name[i]
        # file_label_name = opt + '_' + file_label_name
        files_name[i] = files_name[i]
        file_label_name = file_label_name

        cv2.imwrite(os.path.join(output_images_path, files_name[i]), transformed_image)
        write_bb(os.path.join(output_labels_path, file_label_name), transformed_bboxes, ids)

    cv2.destroyAllWindows()

if __name__ == '__main__':

    #清空输出文件夹
    #delet_images()

    # 从原数据集随机挑选图片，到目标目录
    random_files(500)

    # 获取增强管道
    #augmentors = make_augmentor()

    # 进行图片增强
    #augment_images(augmentors)





