# -*- coding: utf-8 -*-
import sys

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton


class Ui_MainWindow(QMainWindow):
    # 定义定时器，以及按钮状态设置
    def __init__(self) -> None:
        super().__init__()

        '''自定义部分'''
        self.my_timer = QTimer()  # 创建定时器
        self.my_timer.timeout.connect(self.my_timer_cb)  # 创建定时器任务

        '''按钮状态控制'''
        self.btn_status = False

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(860, 542)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icon/icon.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 14, 640, 470))
        self.label.setMouseTracking(False)
        self.label.setTabletTracking(False)
        self.label.setAcceptDrops(False)
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label.setScaledContents(False)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setWordWrap(False)
        self.label.setIndent(-1)
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(680, 350, 160, 60))
        self.pushButton.setIconSize(QtCore.QSize(20, 20))
        self.pushButton.setObjectName("pushButton")

        '''按钮监听事件'''
        self.pushButton.clicked.connect(self.btn_start)

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(680, 430, 160, 60))
        self.pushButton_2.setObjectName("pushButton_2")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(680, 20, 160, 280))
        self.textBrowser.setObjectName("textBrowser")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 860, 26))
        self.menuBar.setObjectName("menuBar")
        self.menu = QtWidgets.QMenu(self.menuBar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menuBar)
        self.menuBar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ASL_Detection - v1.0"))
        self.label.setText(_translate("MainWindow", "摄像头"))
        self.pushButton.setText(_translate("MainWindow", "开始"))
        self.pushButton_2.setText(_translate("MainWindow", "退出"))
        self.menu.setTitle(_translate("MainWindow", "关于"))

    #启动和关闭摄像头
    def btn_start(self):
        # 开关
        if self.btn_status:
            self.btn_status = False
        else:
            self.btn_status = True

        #点击按钮：暂停
        if self.btn_status:
            self.pushButton.setText('暂停')
            self.my_timer.start(40)  # 25fps
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # start camera
        else:
        #点击按钮：开始
            self.pushButton.setText('开始')
            self.label.clear()  # 清楚label内容
            self.my_timer.stop()  # 停止定时器
            self.cap.release()  # 关闭摄像头入代码片

    # 完成摄像头数据捕获和基本处理
    def my_timer_cb(self):
        if self.cap:
            """图像获取"""
            ret, self.image = self.cap.read()
            show = cv2.resize(self.image,(640,480))
            show = cv2.flip(show, 1)
            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)

            """图像处理"""

            """处理结果存储"""

            """结果呈现"""
            showImage = QImage(show.data, show.shape[1],show.shape[0],QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(showImage))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()

# 调用方法
    ui.setupUi(MainWindow)
    ui.btn_start()
    ui.my_timer_cb()

    MainWindow.show()
    sys.exit(app.exec_())


