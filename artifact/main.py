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
from Eyem import Eyem_test_Window
from Musc import Musc_test_Window
from Tongue import Tongue_test_Window
from Jaw import Jaw_test_Window
from Forehead import Forehead_test_Window


class NewWindow(QDialog):
    def __init__(self):
        super(NewWindow, self).__init__()
        self.setWindowTitle("伪迹采集系统")
        # desktop = QApplication.desktop()
        # rect = desktop.frameSize()
        # self.resize(QtCore.QSize(rect.width(), rect.height()))
        self.resize(2600, 1600)
        layout = QtWidgets.QVBoxLayout(self)

        # 创建显示倒计时的 QLabel
        self.timer_label = QLabel(self)
        self.timer_label.setAlignment(Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(35)  # 设置字体大小
        self.timer_label.setFont(font)
        layout.addWidget(self.timer_label, alignment=Qt.AlignTop)

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

        self.musc_button = QtWidgets.QPushButton("肌动测试", self)
        self.musc_button.clicked.connect(self.musc_test)
        layout.addWidget(self.musc_button, alignment=Qt.AlignCenter)
        self.musc_button.setMinimumSize(QtCore.QSize(1000, 100))
        self.musc_button.setFont(button_font)

        self.tongue_button = QtWidgets.QPushButton("舌动测试", self)
        self.tongue_button.clicked.connect(self.tongue_test)
        layout.addWidget(self.tongue_button, alignment=Qt.AlignCenter)
        self.tongue_button.setMinimumSize(QtCore.QSize(1000, 100))
        self.tongue_button.setFont(button_font)

        self.jaw_button = QtWidgets.QPushButton("下颚测试", self)
        self.jaw_button.clicked.connect(self.jaw_test)
        layout.addWidget(self.jaw_button, alignment=Qt.AlignCenter)
        self.jaw_button.setMinimumSize(QtCore.QSize(1000, 100))
        self.jaw_button.setFont(button_font)

        self.forehead_button = QtWidgets.QPushButton("上额测试", self)
        self.forehead_button.clicked.connect(self.forehead_test)
        layout.addWidget(self.forehead_button, alignment=Qt.AlignCenter)
        self.forehead_button.setMinimumSize(QtCore.QSize(1000, 100))
        self.forehead_button.setFont(button_font)

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
        self.start_timer(3, self.show_eyem_test)

    def show_eyem_test(self):
        test = Eyem_test_Window()
        test.start_dot_animation()
        test.exec()

    def musc_test(self):
        self.start_timer(3, self.show_musc_test)

    def show_musc_test(self):
        test = Musc_test_Window()
        test.reminder()
        test.exec()

    def tongue_test(self):
        self.start_timer(3, self.show_tongue_test)

    def show_tongue_test(self):
        test = Tongue_test_Window()
        test.reminder_3times()
        test.exec()

    def jaw_test(self):
        self.start_timer(3, self.show_jaw_test)

    def show_jaw_test(self):
        test = Jaw_test_Window()
        test.reminder()
        test.exec()

    def forehead_test(self):
        self.start_timer(3, self.show_forehead_test)

    def show_forehead_test(self):
        test = Forehead_test_Window()
        test.reminder()
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
        self.mainWindow = QDialog()
        self.filepath = None
        self.pushButton_2.clicked.connect(MainWindow.close)  # type: ignore
        self.pushButton.clicked.connect(self.collect)  # type: ignore
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


