import torch
import numpy as np
import cv2

#----------主程序----------

# Model
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='E:/Python Project/yolov5-master/runs/train/exp4/weights/best.pt', force_reload=True)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# OpenCV连接摄像头
# 参数0-打开笔记本的内置摄像头
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    results = model(frame)

    cv2.imshow('ASL_Detection', np.squeeze(results.render()))

    detect_name = results.pandas().xyxy[0]

    print(detect_name['name'])

    # 按q退出
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# 释放
cap.release()
cv2.destroyAllWindows()

#1.7.4