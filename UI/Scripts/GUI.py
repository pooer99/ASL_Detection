# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import torch

# Form implementation generated from reading ui file 'GUI_style2.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget


class Ui_MainWindow(QWidget):
    def __init__(self,mainWindow,application) -> None:
        super().__init__()

        '''初始化参数'''
        self.app = application
        self.my_timer = QTimer()  # 创建定时器
        self.my_timer.timeout.connect(self.opencv_timer)  # 创建定时器任务
        self.camera_btn_status = False  # 按钮状态
        self.camera_detect_status = False # 是否开启摄像头的检测
        self.is_camera_open = False  # 摄像头是否已经开启
        self.is_image_open = False  # 图片是否已经打开
        self.choose_camera = 0 #（0：系统默认摄像头；1：外接摄像头）

        '''引入yolov5训练模型'''
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                               path='E:/Python Project/yolov5-master/runs/train/exp/weights/best.pt')

        '''设置模型参数'''
        self.model.conf = 0.05 # 设置置信度阈值
        self.model.max_det = 1 # 设置最大检测数量

        '''加载UI到窗口'''
        self.setupUi(mainWindow)

        '''按钮监听'''
        self.select_img_btn.clicked.connect(self.select_image_btn)
        self.detect_img_btn.clicked.connect(self.delect_image_btn)
        self.start_camera_btn.clicked.connect(self.camera_btn)
        self.detect_camera_btn.clicked.connect(self.open_detect_btn)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(949, 777)
        MainWindow.setStyleSheet("")
        MainWindow.setDocumentMode(False)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        MainWindow.setFixedSize(MainWindow.width(), MainWindow.height())# 固定尺寸，禁止界面缩放
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.detect_frame = QtWidgets.QFrame(self.centralwidget)
        self.detect_frame.setGeometry(QtCore.QRect(10, 10, 681, 541))
        self.detect_frame.setFrameShape(QtWidgets.QFrame.Box)
        self.detect_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.detect_frame.setObjectName("detect_frame")
        self.detect_window = QtWidgets.QLabel(self.detect_frame)
        self.detect_window.setGeometry(QtCore.QRect(20, 40, 640, 480))
        self.detect_window.setStyleSheet("QLabel{\n"
                                         "    background-color: rgb(0, 0, 0);\n"
                                         "}")
        self.detect_window.setFrameShape(QtWidgets.QFrame.Box)
        self.detect_window.setFrameShadow(QtWidgets.QFrame.Raised)
        self.detect_window.setText("")
        self.detect_window.setTextFormat(QtCore.Qt.AutoText)
        self.detect_window.setScaledContents(False)
        self.detect_window.setAlignment(QtCore.Qt.AlignCenter)
        self.detect_window.setObjectName("detect_window")
        self.label = QtWidgets.QLabel(self.detect_frame)
        self.label.setGeometry(QtCore.QRect(20, 10, 101, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setObjectName("label")
        self.img_frame = QtWidgets.QFrame(self.centralwidget)
        self.img_frame.setGeometry(QtCore.QRect(700, 50, 231, 211))
        self.img_frame.setFrameShape(QtWidgets.QFrame.Box)
        self.img_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.img_frame.setObjectName("img_frame")
        self.select_img_btn = QtWidgets.QPushButton(self.img_frame)
        self.select_img_btn.setGeometry(QtCore.QRect(30, 50, 161, 41))
        self.select_img_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.select_img_btn.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.select_img_btn.setAutoFillBackground(False)
        self.select_img_btn.setStyleSheet("QPushButton {\n"
                                          "    background-color: #ffffff;\n"
                                          "    border: 1px solid rgb(76, 76, 76);\n"
                                          "    padding: 10px;\n"
                                          "    border-radius: 5px;\n"
                                          "}\n"
                                          "\n"
                                          "QPushButton:hover {\n"
                                          "    background-color: #ecf5ff;\n"
                                          "    color: #409eff;\n"
                                          "}\n"
                                          "\n"
                                          "QPushButton:pressed, QPushButton:checked {\n"
                                          "    border: 1px solid #3a8ee6;\n"
                                          "    color: #409eff;\n"
                                          "}\n"
                                          "\n"
                                          "#button3 {\n"
                                          "    border-radius: 20px;\n"
                                          "}")
        self.select_img_btn.setAutoDefault(False)
        self.select_img_btn.setDefault(False)
        self.select_img_btn.setObjectName("select_img_btn")
        self.detect_img_btn = QtWidgets.QPushButton(self.img_frame)
        self.detect_img_btn.setGeometry(QtCore.QRect(30, 130, 161, 41))
        self.detect_img_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.detect_img_btn.setStyleSheet("QPushButton {\n"
                                          "    background-color: #ffffff;\n"
                                          "    border: 1px solid rgb(76, 76, 76);\n"
                                          "    padding: 10px;\n"
                                          "    border-radius: 5px;\n"
                                          "}\n"
                                          "\n"
                                          "QPushButton:hover {\n"
                                          "    background-color: #ecf5ff;\n"
                                          "    color: #409eff;\n"
                                          "}\n"
                                          "\n"
                                          "QPushButton:pressed, QPushButton:checked {\n"
                                          "    border: 1px solid #3a8ee6;\n"
                                          "    color: #409eff;\n"
                                          "}\n"
                                          "\n"
                                          "#button3 {\n"
                                          "    border-radius: 20px;\n"
                                          "}")
        self.detect_img_btn.setAutoDefault(False)
        self.detect_img_btn.setDefault(False)
        self.detect_img_btn.setObjectName("detect_img_btn")
        self.camera_frame = QtWidgets.QFrame(self.centralwidget)
        self.camera_frame.setGeometry(QtCore.QRect(700, 330, 231, 221))
        self.camera_frame.setFrameShape(QtWidgets.QFrame.Box)
        self.camera_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.camera_frame.setObjectName("camera_frame")
        self.start_camera_btn = QtWidgets.QPushButton(self.camera_frame)
        self.start_camera_btn.setGeometry(QtCore.QRect(30, 60, 161, 41))
        self.start_camera_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.start_camera_btn.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.start_camera_btn.setAutoFillBackground(False)
        self.start_camera_btn.setStyleSheet("QPushButton {\n"
                                            "    background-color: #ffffff;\n"
                                            "    border: 1px solid rgb(76, 76, 76);\n"
                                            "    padding: 10px;\n"
                                            "    border-radius: 5px;\n"
                                            "}\n"
                                            "\n"
                                            "QPushButton:hover {\n"
                                            "    background-color: #ecf5ff;\n"
                                            "    color: #409eff;\n"
                                            "}\n"
                                            "\n"
                                            "QPushButton:pressed, QPushButton:checked {\n"
                                            "    border: 1px solid #3a8ee6;\n"
                                            "    color: #409eff;\n"
                                            "}\n"
                                            "\n"
                                            "#button3 {\n"
                                            "    border-radius: 20px;\n"
                                            "}")
        self.start_camera_btn.setAutoDefault(False)
        self.start_camera_btn.setDefault(False)
        self.start_camera_btn.setObjectName("start_camera_btn")
        self.detect_camera_btn = QtWidgets.QPushButton(self.camera_frame)
        self.detect_camera_btn.setGeometry(QtCore.QRect(30, 140, 161, 41))
        self.detect_camera_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.detect_camera_btn.setStyleSheet("QPushButton {\n"
                                             "    background-color: #ffffff;\n"
                                             "    border: 1px solid rgb(76, 76, 76);\n"
                                             "    padding: 10px;\n"
                                             "    border-radius: 5px;\n"
                                             "}\n"
                                             "\n"
                                             "QPushButton:hover {\n"
                                             "    background-color: #ecf5ff;\n"
                                             "    color: #409eff;\n"
                                             "}\n"
                                             "\n"
                                             "QPushButton:pressed, QPushButton:checked {\n"
                                             "    border: 1px solid #3a8ee6;\n"
                                             "    color: #409eff;\n"
                                             "}\n"
                                             "\n"
                                             "#button3 {\n"
                                             "    border-radius: 20px;\n"
                                             "}")
        self.detect_camera_btn.setAutoDefault(False)
        self.detect_camera_btn.setDefault(False)
        self.detect_camera_btn.setObjectName("detect_camera_btn")
        self.detect_print_title = QtWidgets.QLabel(self.centralwidget)
        self.detect_print_title.setGeometry(QtCore.QRect(10, 560, 72, 41))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.detect_print_title.setFont(font)
        self.detect_print_title.setObjectName("detect_print_title")
        self.detect_print_text = QtWidgets.QLabel(self.centralwidget)
        self.detect_print_text.setGeometry(QtCore.QRect(10, 600, 671, 121))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.detect_print_text.setFont(font)
        self.detect_print_text.setText("")
        self.detect_print_text.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.detect_print_text.setObjectName("detect_print_text")
        self.detect_img_title = QtWidgets.QLabel(self.centralwidget)
        self.detect_img_title.setGeometry(QtCore.QRect(700, 0, 72, 41))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.detect_img_title.setFont(font)
        self.detect_img_title.setObjectName("detect_img_title")
        self.detect_camera_title = QtWidgets.QLabel(self.centralwidget)
        self.detect_camera_title.setGeometry(QtCore.QRect(700, 280, 72, 41))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.detect_camera_title.setFont(font)
        self.detect_camera_title.setObjectName("detect_camera_title")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 949, 26))
        self.menuBar.setObjectName("menuBar")
        self.menu = QtWidgets.QMenu(self.menuBar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menuBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.menuBar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ASL_Detection - v1.5"))
        MainWindow.setWindowIcon(QIcon('../Resources/icon.ico'))

        self.label.setText(_translate("MainWindow", "检测窗口:"))
        self.select_img_btn.setText(_translate("MainWindow", "选择图片"))
        self.detect_img_btn.setText(_translate("MainWindow", "检测"))
        self.start_camera_btn.setText(_translate("MainWindow", "开启摄像头"))
        self.detect_camera_btn.setText(_translate("MainWindow", "开启检测"))
        self.detect_print_title.setText(_translate("MainWindow", "检测信息："))
        self.detect_print_text.setText(_translate("MainWindow", ""))
        self.detect_img_title.setText(_translate("MainWindow", "图片检测："))
        self.detect_camera_title.setText(_translate("MainWindow", "实时检测："))
        self.menu.setTitle(_translate("MainWindow", "关于"))

    '''输出检测信息'''
    def print_detect_info(self, pd):
        # 检测信息
        detect_pd = pd.xyxy[0].sort_values('confidence', ascending=False)  # 按准确度降序排序
        detect_name = detect_pd['name'].to_numpy()
        detect_confidence = detect_pd['confidence'].to_numpy()

        # 打印信息
        info = '检测对象数量：' + str(len(detect_name)) + '\n'
        if len(detect_name) >= 1:
            for i in range(len(detect_name)):
                info = info + "检测对象：" + str(detect_name[i]) + ", 准确度：" + str(detect_confidence[i]) + '\n'

        # 显示检测信息
        self.detect_print_text.setText(info)

    '''选择本地图片'''
    def select_image_btn(self):
        # 关闭摄像头
        if self.is_camera_open :
            self.is_camera_open = False
            self.camera_btn_status = False
            self.camera_detect_status = False
            self.detect_print_text.clear()
            self.start_camera_btn.setText('开启摄像头')
            self.detect_window.clear()  # 清除检测窗口内容
            self.detect_print_text.clear()  # 清除检测信息内容
            self.my_timer.stop()  # 停止定时器
            self.cap.release()  # 关闭摄像头入代码片

        # 选择图片
        self.is_image_open = True
        self.image, _ = QFileDialog.getOpenFileName(self, '打开文件', './', '图像文件(*.jpg *.png)')

        if self.image is not None:
            # 缩放图片，自适应窗口
            tran_image = QImage(self.image).scaled(640,480,Qt.IgnoreAspectRatio)
            # 显示图片
            self.detect_window.setPixmap(QPixmap.fromImage(tran_image))

    '''检测本地图片'''
    def delect_image_btn(self):
        if self.is_image_open :
            if self.image is not None:
                results = self.model(self.image)

                # 渲染后的图片
                img = np.squeeze(results.render())

                # 打印检测信息
                self.print_detect_info(results.pandas())

                # 显示图片
                x = img.shape[1]
                y = img.shape[0]
                show_image = QImage(img.data, x, y, x * 3, QImage.Format_RGB888)

                # 缩放图片，自适应窗口
                show_image = show_image.scaled(640, 480, Qt.IgnoreAspectRatio)
                # 显示检测后的图片
                self.detect_window.setPixmap(QPixmap.fromImage(show_image))

    '''检测摄像头'''
    # 开启和关闭摄像头 按钮事件
    def camera_btn(self):
        # 开关
        if self.camera_btn_status:
            self.camera_btn_status = False
        else:
            self.camera_btn_status = True

        # 开启摄像头
        if self.camera_btn_status:
            # 清空图片
            self.is_image_open = False
            self.detect_print_text.clear()
            self.detect_window.clear()

            self.is_camera_open = True
            self.start_camera_btn.setText('关闭摄像头')
            self.my_timer.start(40)  # 25fps
            self.cap = cv2.VideoCapture(self.choose_camera, cv2.CAP_DSHOW)  # 开启摄像头（0：系统默认摄像头；1：外接摄像头）
        else:
            # 停止
            self.is_camera_open = False
            self.start_camera_btn.setText('开启摄像头')
            self.detect_window.clear()  # 清除检测窗口内容
            self.detect_print_text.clear()  # 清除检测信息内容
            self.my_timer.stop()  # 停止定时器
            self.camera_detect_status = False
            self.cap.release()  # 关闭摄像头入代码片

        # 完成摄像头数据捕获和基本处理

    # 开启检测 按钮事件
    def open_detect_btn(self):
        if self.is_camera_open:
            # 开关
            if self.camera_detect_status:
                self.camera_detect_status = False
                self.detect_print_text.clear()
            else:
                self.camera_detect_status = True

    # 使用openCV和yolov5检测对象
    def opencv_timer(self):
        if self.cap:
            """图像获取"""
            ret, frame = self.cap.read()
            show = cv2.resize(frame, (640, 480))
            show = cv2.flip(show, 1)
            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)

            # 检测
            img = show
            if self.camera_detect_status :
                """图像处理"""
                results = self.model(show)

                '''获取检测结果信息'''
                # 渲染后的图片
                img = np.squeeze(results.render())

                # 打印信息
                self.print_detect_info(results.pandas())

                """结果呈现"""
                # 显示摄像头
                show_image = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
                self.detect_window.setPixmap(QPixmap.fromImage(show_image))

            # 不检测
            show_image = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
            self.detect_window.setPixmap(QPixmap.fromImage(show_image))

if __name__ == '__main__':
    app = QApplication(sys.argv)

    MainWindow = QMainWindow()
    ui = Ui_MainWindow(MainWindow,app)
    MainWindow.show()

    sys.exit(app.exec_())
