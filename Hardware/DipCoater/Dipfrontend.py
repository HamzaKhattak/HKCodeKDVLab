# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Dipfrontend.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(603, 699)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(20, 10, 563, 635))
        self.widget.setObjectName("widget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_5 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.verticalLayout.addWidget(self.label_5)
        self.label_6 = QtWidgets.QLabel(self.widget)
        self.label_6.setWordWrap(True)
        self.label_6.setObjectName("label_6")
        self.verticalLayout.addWidget(self.label_6)
        self.verticalLayout_3.addLayout(self.verticalLayout)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_13 = QtWidgets.QLabel(self.widget)
        self.label_13.setObjectName("label_13")
        self.gridLayout_2.addWidget(self.label_13, 0, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.widget)
        self.label_10.setObjectName("label_10")
        self.gridLayout_2.addWidget(self.label_10, 0, 1, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.widget)
        self.label_11.setObjectName("label_11")
        self.gridLayout_2.addWidget(self.label_11, 0, 2, 1, 1)
        self.controller_COMport = QtWidgets.QLineEdit(self.widget)
        self.controller_COMport.setObjectName("controller_COMport")
        self.gridLayout_2.addWidget(self.controller_COMport, 1, 0, 1, 1)
        self.controller_pulserev = QtWidgets.QLineEdit(self.widget)
        self.controller_pulserev.setObjectName("controller_pulserev")
        self.gridLayout_2.addWidget(self.controller_pulserev, 1, 1, 1, 1)
        self.distance_per_rotation = QtWidgets.QLineEdit(self.widget)
        self.distance_per_rotation.setObjectName("distance_per_rotation")
        self.gridLayout_2.addWidget(self.distance_per_rotation, 1, 2, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout_2)
        self.param_set = QtWidgets.QPushButton(self.widget)
        self.param_set.setObjectName("param_set")
        self.horizontalLayout.addWidget(self.param_set)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.label_8 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_3.addWidget(self.label_8)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setKerning(True)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 2, 1, 1)
        self.main_speed = QtWidgets.QLineEdit(self.widget)
        self.main_speed.setObjectName("main_speed")
        self.gridLayout.addWidget(self.main_speed, 1, 0, 1, 1)
        self.main_acceleration = QtWidgets.QLineEdit(self.widget)
        self.main_acceleration.setObjectName("main_acceleration")
        self.gridLayout.addWidget(self.main_acceleration, 1, 1, 1, 1)
        self.main_distance = QtWidgets.QLineEdit(self.widget)
        self.main_distance.setObjectName("main_distance")
        self.gridLayout.addWidget(self.main_distance, 1, 2, 1, 1)
        self.horizontalLayout_2.addLayout(self.gridLayout)
        self.main_start = QtWidgets.QPushButton(self.widget)
        self.main_start.setObjectName("main_start")
        self.horizontalLayout_2.addWidget(self.main_start)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.line = QtWidgets.QFrame(self.widget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_3.addWidget(self.line)
        self.label_7 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_3.addWidget(self.label_7)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 0, 0, 1, 1)
        self.jogup = QtWidgets.QPushButton(self.widget)
        self.jogup.setObjectName("jogup")
        self.gridLayout_3.addWidget(self.jogup, 0, 2, 1, 1)
        self.jogspeed = QtWidgets.QLineEdit(self.widget)
        self.jogspeed.setObjectName("jogspeed")
        self.gridLayout_3.addWidget(self.jogspeed, 1, 0, 1, 1)
        self.jogdown = QtWidgets.QPushButton(self.widget)
        self.jogdown.setObjectName("jogdown")
        self.gridLayout_3.addWidget(self.jogdown, 1, 2, 1, 1)
        self.jogtimestep = QtWidgets.QLineEdit(self.widget)
        self.jogtimestep.setObjectName("jogtimestep")
        self.gridLayout_3.addWidget(self.jogtimestep, 1, 1, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.widget)
        self.label_14.setObjectName("label_14")
        self.gridLayout_3.addWidget(self.label_14, 0, 1, 1, 1)
        self.verticalLayout_3.addLayout(self.gridLayout_3)
        self.line_2 = QtWidgets.QFrame(self.widget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout_3.addWidget(self.line_2)
        self.label_9 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.verticalLayout_3.addWidget(self.label_9)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setObjectName("label_4")
        self.gridLayout_4.addWidget(self.label_4, 0, 0, 1, 1)
        self.csvstart = QtWidgets.QPushButton(self.widget)
        self.csvstart.setObjectName("csvstart")
        self.gridLayout_4.addWidget(self.csvstart, 1, 1, 1, 1)
        self.csvaddress = QtWidgets.QLineEdit(self.widget)
        self.csvaddress.setObjectName("csvaddress")
        self.gridLayout_4.addWidget(self.csvaddress, 1, 0, 1, 1)
        self.verticalLayout_3.addLayout(self.gridLayout_4)
        self.line_3 = QtWidgets.QFrame(self.widget)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.verticalLayout_3.addWidget(self.line_3)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_15 = QtWidgets.QLabel(self.widget)
        self.label_15.setObjectName("label_15")
        self.verticalLayout_2.addWidget(self.label_15)
        self.approxpos = QtWidgets.QLCDNumber(self.widget)
        self.approxpos.setObjectName("approxpos")
        self.verticalLayout_2.addWidget(self.approxpos)
        self.horizontalLayout_3.addLayout(self.verticalLayout_2)
        self.text_monitor = QtWidgets.QTextBrowser(self.widget)
        self.text_monitor.setObjectName("text_monitor")
        self.horizontalLayout_3.addWidget(self.text_monitor)
        self.butStop = QtWidgets.QPushButton(self.widget)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.butStop.setFont(font)
        self.butStop.setObjectName("butStop")
        self.horizontalLayout_3.addWidget(self.butStop)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 603, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Chip"))
        self.label_5.setText(_translate("MainWindow", "Chip the Dip Coater and Fiber Puller Controller"))
        self.label_6.setText(_translate("MainWindow", "This is the GUI to control the DIY dipcoater/fiber puller named Chip set up in the lab."))
        self.label_13.setText(_translate("MainWindow", "COM port"))
        self.label_10.setText(_translate("MainWindow", "Controller pulse/revolution"))
        self.label_11.setText(_translate("MainWindow", "Distance per rotation (mm)"))
        self.param_set.setText(_translate("MainWindow", "Set"))
        self.label_8.setText(_translate("MainWindow", "Speed + acceleration + distance"))
        self.label.setText(_translate("MainWindow", "Speed (mm/s)"))
        self.label_12.setText(_translate("MainWindow", "<html><head/><body><p>Acceleration (mm<span style=\" vertical-align:super;\">2</span>/s)</p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "Distance (mm)"))
        self.main_start.setText(_translate("MainWindow", "Start"))
        self.label_7.setText(_translate("MainWindow", "Jog"))
        self.label_3.setText(_translate("MainWindow", "Speed (mm/s)"))
        self.jogup.setText(_translate("MainWindow", "Up"))
        self.jogdown.setText(_translate("MainWindow", "Down"))
        self.label_14.setText(_translate("MainWindow", "Time step for jog (ms)"))
        self.label_9.setText(_translate("MainWindow", "csv speeds+times (not yet working)"))
        self.label_4.setText(_translate("MainWindow", "File Address"))
        self.csvstart.setText(_translate("MainWindow", "Start"))
        self.label_15.setText(_translate("MainWindow", "Position (mm)"))
        self.butStop.setText(_translate("MainWindow", "STOP"))

