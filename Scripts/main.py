import torch
import numpy as np
import cv2

#----------主程序----------

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# OpenCV连接摄像头
# 参数0-打开笔记本的内置摄像头
cap = cv2.VideoCapture(0)
# 检测置信度0.5，跟踪置信度0.5
while cap.isOpened():
    # ret为bool类型，指示是否成功读取这一帧
    # 获取摄像头帧
    ret, frame = cap.read()

    # 对帧进行检测并返回检测结果results
    results = model(frame)

    # 绘制骨骼结点

    # 预测

    # 显示在窗口中
    cv2.imshow('ASL_Detection', np.squeeze(results.render()))

    # 按q退出
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


# 释放
cap.release()
cv2.destroyAllWindows()

#31.29