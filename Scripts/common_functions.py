import os
from os import path
import glob
import cv2
import albumentations as A
import matplotlib.pyplot as plt

# read image yolo labels from yolo file
# and returns in yolo format
def read_bb(label_path):
    with open(label_path, "r") as f:
        data = f.readlines()

    # build the bb list
    yolo_bb_list = []

    for line in data:
        # Split string to float
        class_num, x, y, w, h = map(float, line.split(" "))

        # in the list, class_num is the last, ready for Albumentations
        yolo_bb_list.append([x, y, w, h, int(class_num)])

    return yolo_bb_list


# write in a file the transformed bb
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


# convert a single bb for cv2
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

def read_images(path):
    names = os.listdir(path)  # 获取images文件名
    images = []  # 完整路径

    # 生成路径
    for i in range(len(names)):
        images.append(os.path.join(path, names[i]))

    return names, images


