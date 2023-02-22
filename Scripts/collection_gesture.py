import uuid
import time
import numpy as np
import cv2
import os
import config as cf

#----------收集训练素材，打标签----------

for i in range(len(cf.labels)):
    file_path = os.path.join(cf.IMAGE_PATH, cf.labels[i])
    print(cf.labels[i] + ': ')
    print(os.listdir(file_path))