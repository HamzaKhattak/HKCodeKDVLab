# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DipGUI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

import serial
import math
import time
import numpy

#The GUI

class infoSender:
    def serializecsv(self, parameter_list):
        pass
    def sendcsv(self, filetosend):
        pass

class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(480, 640)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(10, 210, 441, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(10, 340, 441, 20))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(10, 460, 441, 20))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.butStop = QtWidgets.QPushButton(self.centralwidget)
        self.butStop.setGeometry(QtCore.QRect(330, 500, 111, 71))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.butStop.setFont(font)
        self.butStop.setObjectName("butStop")
        self.jogdown = QtWidgets.QPushButton(self.centralwidget)
        self.jogdown.setGeometry(QtCore.QRect(370, 310, 75, 24))
        self.jogdown.setObjectName("jogdown")
        self.jogup = QtWidgets.QPushButton(self.centralwidget)
        self.jogup.setGeometry(QtCore.QRect(370, 270, 75, 24))
        self.jogup.setObjectName("jogup")
        self.csvstart = QtWidgets.QPushButton(self.centralwidget)
        self.csvstart.setGeometry(QtCore.QRect(370, 430, 75, 24))
        self.csvstart.setObjectName("csvstart")
        self.speedtime_start = QtWidgets.QPushButton(self.centralwidget)
        self.speedtime_start.setGeometry(QtCore.QRect(370, 180, 75, 24))
        self.speedtime_start.setObjectName("speedtime_start")
        self.speedtime_speed = QtWidgets.QLineEdit(self.centralwidget)
        self.speedtime_speed.setGeometry(QtCore.QRect(40, 180, 113, 22))
        self.speedtime_speed.setObjectName("speedtime_speed")
        self.csvaddress = QtWidgets.QLineEdit(self.centralwidget)
        self.csvaddress.setGeometry(QtCore.QRect(40, 430, 261, 22))
        self.csvaddress.setObjectName("csvaddress")
        self.jogspeed = QtWidgets.QLineEdit(self.centralwidget)
        self.jogspeed.setGeometry(QtCore.QRect(40, 290, 113, 22))
        self.jogspeed.setObjectName("jogspeed")
        self.status_icon = QtWidgets.QGraphicsView(self.centralwidget)
        self.status_icon.setGeometry(QtCore.QRect(200, 500, 71, 61))
        self.status_icon.setObjectName("status_icon")
        self.speedtime_time = QtWidgets.QLineEdit(self.centralwidget)
        self.speedtime_time.setGeometry(QtCore.QRect(190, 180, 113, 22))
        self.speedtime_time.setObjectName("speedtime_time")
        self.approxpos = QtWidgets.QLCDNumber(self.centralwidget)
        self.approxpos.setGeometry(QtCore.QRect(50, 520, 64, 23))
        self.approxpos.setObjectName("approxpos")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 160, 101, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(190, 160, 55, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(40, 270, 121, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(40, 410, 131, 16))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(20, 10, 471, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(50, 50, 361, 51))
        self.label_6.setWordWrap(True)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(200, 230, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(170, 120, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(100, 360, 341, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 480, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        '''

        self.jogdown.clicked.connect(self.jogdownfx)
        self.jogup.clicked.connect(self.jogupfx)
        self.ststart.clicked.connect(self.ststartfx)
        self.csvstart.clicked.connect(self.csvstartfx)
        '''
        #Relevent variables


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.butStop.setText(_translate("MainWindow", "STOP"))
        self.jogdown.setText(_translate("MainWindow", "Down"))
        self.jogup.setText(_translate("MainWindow", "Up"))
        self.csvstart.setText(_translate("MainWindow", "Start"))
        self.speedtime_start.setText(_translate("MainWindow", "Start"))
        self.label.setText(_translate("MainWindow", "Speed (mm/s)"))
        self.label_2.setText(_translate("MainWindow", "Time (s)"))
        self.label_3.setText(_translate("MainWindow", "Speed (mm/s)"))
        self.label_4.setText(_translate("MainWindow", "File Address"))
        self.label_5.setText(_translate("MainWindow", "Chip the Dip Coater and Fiber Puller Controller"))
        self.label_6.setText(_translate("MainWindow", "This is the GUI to control the DIY dipcoater/fiber puller named Chip set up in the lab."))
        self.label_7.setText(_translate("MainWindow", "Jog"))
        self.label_8.setText(_translate("MainWindow", "Speed + time"))
        self.label_9.setText(_translate("MainWindow", "csv speeds+times (not yet working)"))

        #speed time version
        stspeed = 0
        sttime = 0
        #speed 
        jogmoveSpeed = 0
        jogDir = 1
    
    def jogdownfx(self):
        pass
    def jogupfx(self):
        pass
    def csvstartfx(self):
        pass

    def ststartfx(self):
        speedval=self.jogspeed.text()
        print(speedval)

#Need to run in console
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
