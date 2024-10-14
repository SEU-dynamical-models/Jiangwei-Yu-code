# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hello.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QLabel, QSpacerItem, QSizePolicy
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QTimer, QTime, Qt
from PyQt5.QtMultimedia import QSound


class Eyem_test_Window(QDialog):
    def __init__(self):
        super(Eyem_test_Window, self).__init__()
        self.setWindowTitle("眼动伪迹采集")
        self.resize(2600,1600)

        # 创建显示倒计时的 QLabel
        layout = QtWidgets.QVBoxLayout(self)  # 创建一个 QVBoxLayout 布局管理器
        self.timer_label = QLabel(self)
        self.timer_label.setAlignment(Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(35)  # 设置字体大小
        self.timer_label.setFont(font)
        layout.addWidget(self.timer_label, alignment=Qt.AlignCenter)

        # 创建定时器
        self.dot_timer = QTimer(self)
        self.rest_timer = QTimer(self)
        self.dot_timer.timeout.connect(self.show_dot)
        self.rest_timer.timeout.connect(self.rest_update)
        self.dot_visible = True  # 红点是否可见
        self.dot_position = 0  # 控制红点的位置（0 为顶部/左部，1 为底部/右部）
        self.is_dot_centered = False  # 标志位，是否已经将红点放置到窗口中心

    def rest(self):
        self.is_dot_centered = False
        self.update()
        self.remaintime = 5
        self.rest_timer.start(1000)

    def eyeblink(self):
        self.round = 2
        self.is_dot_centered = False
        self.update()
        self.remaintime = 5
        self.rest_timer.start(1000)

    def rest_update(self):
        # 更新显示的时间

        minutes = self.remaintime // 60
        seconds = self.remaintime % 60
        label_text = "休息时间"
        if self.round == 2 :
            label_text = "正常眨眼"
        self.timer_label.setText(f"{label_text}:{minutes:02d}:{seconds:02d}")

        # 如果时间到了，停止计时器
        if self.remaintime <= 0:
            self.rest_timer.stop()
            if self.round == 0:
                self.next_round()
            elif self.round == 1:
                NewWindow().play_alarm()
                self.eyeblink()
            elif self.round == 2:
                NewWindow().play_alarm()
                self.timer_label.setText("眼动测试结束!")
        else:
            self.remaintime -= 1  # 每次减少 1 秒

    def start_dot_animation(self):
        self.verorhor = 0
        self.eyem_times = 10
        self.round = 0
        self.dot_timer.start(1000)  # 每 1 秒更新一次

    def next_round(self):
        self.verorhor = 1
        self.eyem_times = 10
        self.round = 1
        self.dot_timer.start(1000)  # 每 1 秒更新一次

    def show_dot(self):
        if self.eyem_times != 0:
            self.dot_position = 1 - self.dot_position  # 切换位置
            # 触发提示音
            NewWindow().play_alarm()
            # 更新窗口以绘制红点
            self.update()
            # 次数减一
            self.eyem_times -= 1
        else:
            self.dot_timer.stop()
            if self.round == 0 or 1:
                self.rest()


    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # 绘制红点
        if self.dot_visible:
            dot_radius = 50
            dot_color = QtGui.QColor(255, 0, 0)  # 红色
            painter.setBrush(dot_color)

            # 确定红点的位置
            if not self.is_dot_centered:
                # 将红点绘制到窗口的中心
                painter.drawEllipse(self.rect().center().x() - dot_radius / 2,
                                    self.rect().center().y() - dot_radius / 2,
                                    dot_radius, dot_radius)
                # 设置标志位为 True，表示已经放置
                self.is_dot_centered = True
            else:
                if self.verorhor == 0:
                    if self.dot_position == 0:  # 顶部
                        painter.drawEllipse(self.rect().center().x() - dot_radius / 2, 0, dot_radius, dot_radius)
                    else:  # 底部
                        painter.drawEllipse(self.rect().center().x() - dot_radius / 2, self.height() - dot_radius,
                                            dot_radius, dot_radius)
                elif self.verorhor == 1:
                    if self.dot_position == 0:  # 左侧
                        painter.drawEllipse(0, self.rect().center().y() - dot_radius / 2, dot_radius, dot_radius)
                    else:  # 右侧
                        painter.drawEllipse(self.width() - dot_radius, self.rect().center().y() - dot_radius / 2,
                                            dot_radius, dot_radius)

class NewWindow(QDialog):
    def __init__(self):
        super(NewWindow, self).__init__()
        self.setWindowTitle("伪迹采集系统")
        # desktop = QApplication.desktop()
        # rect = desktop.frameSize()
        # self.resize(QtCore.QSize(rect.width(), rect.height()))
        self.resize(2600,1600)
        layout = QtWidgets.QVBoxLayout(self)


        # 创建显示倒计时的 QLabel
        self.timer_label = QLabel(self)
        self.timer_label.setAlignment(Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(35)  # 设置字体大小
        self.timer_label.setFont(font)
        layout.addWidget(self.timer_label, alignment= Qt.AlignTop)



        button_font = QtGui.QFont()
        button_font.setPointSize(20)  # 设置字体大小
        layout.addStretch()

        # 创建 QTimer 对象并设置时间间隔为 1 秒（1000 毫秒）
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)

        self.baseline_button = QtWidgets.QPushButton("基线测试", self)
        self.baseline_button.clicked.connect(self.play_alarm)
        self.baseline_button.clicked.connect(lambda: self.start_timer(60))
        layout.addWidget(self.baseline_button, alignment=Qt.AlignCenter)
        self.baseline_button.setMinimumSize(QtCore.QSize(1000, 100))
        self.baseline_button.setFont(button_font)

        self.eyem_button = QtWidgets.QPushButton("眼动测试", self)
        self.eyem_button.clicked.connect(self.eyem_test)
        layout.addWidget(self.eyem_button, alignment=Qt.AlignCenter)
        self.eyem_button.setMinimumSize(QtCore.QSize(1000, 100))
        self.eyem_button.setFont(button_font)


        self.close_button = QtWidgets.QPushButton("关闭", self)
        self.close_button.clicked.connect(self.close)  # 关闭窗口
        layout.addWidget(self.close_button, alignment=Qt.AlignCenter)
        self.close_button.setMinimumSize(QtCore.QSize(1000, 100))
        self.close_button.setFont(button_font)


    def start_timer(self, duration, callback=None):
        self.remaining_time = duration
        self.timer.start(1000)  # 每秒更新一次
        self.update_timer()
        self.callback = callback

    def update_timer(self):
        # 更新显示的时间
        minutes = self.remaining_time // 60
        seconds = self.remaining_time % 60
        self.timer_label.setText(f"{minutes:02d}:{seconds:02d}")

        # 如果时间到了，停止计时器
        if self.remaining_time <= 0:
            self.timer.stop()
            self.timer_label.setText("时间到！")
            self.play_alarm()  # 播放提示音
            if self.callback:
                self.callback()
        else:
            self.remaining_time -= 1  # 每次减少 1 秒

    def play_alarm(self):
        # 播放声音，确保你有这个文件
        QSound.play("alert.wav")  # 音频文件路径

    def eyem_test(self):
        self.start_timer(3,self.show_eyem_test)

    def show_eyem_test(self):
        test = Eyem_test_Window()
        test.start_dot_animation()
        test.exec()




class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1600, 1200)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout.addLayout(self.gridLayout)
        self.pushButton = QtWidgets.QPushButton()
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setMinimumSize(QtCore.QSize(1000, 500))
        self.verticalLayout.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton()
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setMinimumSize(QtCore.QSize(1000, 200))
        self.verticalLayout.addWidget(self.pushButton_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.mainWindow=QDialog()
        self.filepath=None
        self.pushButton_2.clicked.connect(MainWindow.close) # type: ignore
        self.pushButton.clicked.connect(self.collect) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "脑电信号伪迹采集系统"))
        self.pushButton.setText(_translate("MainWindow", "开始采集"))
        font = QtGui.QFont()
        font.setPointSize(20)  # 设置字体大小为20
        self.pushButton.setFont(font)  # 应用字体设置
        self.pushButton_2.setText(_translate("MainWindow", "退出"))
        self.pushButton_2.setFont(font)  # 应用字体设置

    def collect(self):
        new_window = NewWindow()  # 创建新窗口实例
        new_window.exec_()  # 显示窗口

