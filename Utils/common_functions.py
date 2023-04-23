import os
import random
import shutil

import cv2
import matplotlib.pyplot as plt


'''保存label的边框坐标'''
def write_bb(new_label_path, yolo_bb_list, ids_list):
    # a single component in yolo_bb_list is xc, yc, w, h, class_num
    # so we need to change the order in the file (where class_num is the first)
    with open(new_label_path, "w") as f:
        for i in range(len(yolo_bb_list)):
            # get a row

            xc, yc, w, h = yolo_bb_list[i]
            class_num = ids_list[i]

            # of decimal digits: 8
            new_line = f"{class_num} {xc:.8f} {yc:.8f} {w:.8f} {h:.8f}\n"

            f.write(new_line)
    f.close()


'''将yolo格式坐标转换为xyxy，用于cv2绘制边框'''
def yolo_to_cv2(yolo_bb, size):
    # yolo_bb is list o a tuple
    # with the order of field as expected from Albumentations
    # class_num is the last one
    # width, height are the width, height of the entire image
    # w, h are for the BB
    height = size[0]
    width = size[1]

    # the last is the class_num, here not used
    x, y, w, h = yolo_bb

    # x lower left. max to restrict if outside
    # if < 0 then 0
    l = max(0, int((x - w / 2.0) * width))
    # x upper right
    r = min(int((x + w / 2.0) * width), width - 1)
    # y lower left
    t = max(0, int((y - h / 2.0) * height))
    # y upper right
    b = min(int((y + h / 2.0) * height), height - 1)

    return [l, r, t, b]

'''读取原图片路径'''
def read_images(path):
    names = os.listdir(path)  # 获取images文件名
    images = []  # 完整路径

    # 生成路径
    for i in range(len(names)):
        images.append(os.path.join(path, names[i]))

    return names, images

'''定义可视化函数'''
'''显示图片'''
def visualize(image):
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(image)
    plt.waitforbuttonpress()

'''绘制边框'''
def visualize_bbox(output_path, size, img, ids, bboxes, color=(255, 0, 0), thickness=1):
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
    cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_path, image_name), img)

    cv2.destroyAllWindows()
    return img

'''显示图片'''
def show_iamges_with_bboxes(image, bbox):
    result = visualize_bbox(image, bbox)
    visualize(result)


'''从原数据集中随机获取images及其labels'''
def random_files(count):
    # 原始文件夹路径和目标文件夹路径
    src_dir = "E:/Python Project/yolov5/data/test"
    src_images_path = os.path.join(src_dir, 'images')
    src_labels_path = os.path.join(src_dir, 'labels')

    dst_dir = "F:/augment_image_test/input"
    dst_images_path = os.path.join(dst_dir, 'images')
    dst_labels_path = os.path.join(dst_dir, 'labels')

    # 获取原始文件夹中的所有文件名
    file_list = os.listdir(src_images_path)

    # 随机选择200个文件名
    selected_files = random.sample(file_list, count)

    # 复制选中的文件到目标文件夹中
    for file_name in selected_files:
        # 图片复制
        images_src_path = os.path.join(src_images_path, file_name)
        images_dst_path = os.path.join(dst_images_path, file_name)
        shutil.copy(images_src_path, images_dst_path)

        # 标签复制
        label_name = file_name[:-3] + 'txt'
        labels_src_path = os.path.join(src_labels_path, label_name)
        labels_dst_path = os.path.join(dst_labels_path, label_name)
        shutil.copy(labels_src_path, labels_dst_path)

'''删除测试数据'''
def delet_images():
    output_images_path = r'F:\augment_image_test\out\images'
    output_labels_path = r'F:\augment_image_test\out\labels'

    # 清空文件夹
    shutil.rmtree(output_images_path)
    os.mkdir(output_images_path)

    # 重新生成文件夹
    shutil.rmtree(output_labels_path)
    os.mkdir(output_labels_path)

    # 复制classes.txt
    shutil.copy(r'F:\augment_image_test\input\labels\classes.txt', output_labels_path)

'''随机选择一个数据增强方法'''
def random_augment_opt(augs):
    opt = random.choice(list(augs.keys()))
    transform = augs[opt]
    return transform, opt


