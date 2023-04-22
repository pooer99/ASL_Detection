import os
import cv2
import albumentations as A
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


