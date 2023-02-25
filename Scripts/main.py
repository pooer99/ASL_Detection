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

    detect_pd = results.pandas().xyxy[0].sort_values('confidence', ascending=False)  # 按准确度降序排序
    detect_name = detect_pd['name'].to_numpy()
    detect_confidence = detect_pd['confidence'].to_numpy()
    if(len(detect_name)>=1):
        info = "主要检测对象：" + str(detect_name[0]) + ", 准确度：" + str(detect_confidence[0])
        print(info)

    # 按q退出
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# 释放
cap.release()
cv2.destroyAllWindows()

#1.7.4