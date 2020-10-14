# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DipGUI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

import serial
import time
from serial import SerialException
class motioncontol:
    def __init__(self,portval):
        self.ser=serial.Serial(
                port = portval,
                baudrate = 9600)

    def writecommand(self,docuCommand):
        modstring=docuCommand +'\r\n'
        self.ser.write(modstring.encode())
        
    def getinfo(self):
        rawresult = self.ser.readline().decode()
        result = rawresult.strip()
        return 	result

    def closeport(self):
        self.ser.close()
    
    def sendfcommand(self,paramkey,valueint,valuefloat):
        '''
        paramkey is a single character associated with the command
        valueint and valuefloat are what the param needs to be set to
        floats are rounded to 2 decimal places
        '''
        convint = str(valueint)
        convflt = str(round(valuefloat,2))
        writestring = '<'+ paramkey + ',' + convint + ',' + convflt +'>'
        self.writecommand(writestring) 

    def setspeedacc(self,speedmm,accmm,mmperstep):
        #Calculate the motion in steps and convert to strings
        speedstp = round(speedmm / mmperstep,2)
        accstp = round(accmm / mmperstep,2)
        self.sendfcommand('A',0, accstp)
        self.sendfcommand('S',0, speedstp)


    def moverelative(self,distancemm,mmperstep):
        #Speed and acceleration should already be set
        distancestp = int(distancemm / mmperstep)
        self.writecommand('<G,0,0>') #Write position
        current_location_stp = int(self.getinfo())
        newlocstp = int(current_location_stp + distancestp)
        self.sendfcommand('M',newlocstp, 0) #final location should be integer since can't be at half step

    def timejogmove(self,t_in):
        '''
        Simply sends out a job command in a given direction, enter time in seconds
        Speed should be set beforehand
        '''
        self.sendfcommand('J',0,t_in)

        


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(566, 631)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.verticalLayout.addWidget(self.label_5)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setWordWrap(True)
        self.label_6.setObjectName("label_6")
        self.verticalLayout.addWidget(self.label_6)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setObjectName("label_13")
        self.gridLayout_2.addWidget(self.label_13, 0, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setObjectName("label_10")
        self.gridLayout_2.addWidget(self.label_10, 0, 1, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setObjectName("label_11")
        self.gridLayout_2.addWidget(self.label_11, 0, 2, 1, 1)
        self.controller_COMport = QtWidgets.QLineEdit(self.centralwidget)
        self.controller_COMport.setObjectName("controller_COMport")
        self.gridLayout_2.addWidget(self.controller_COMport, 1, 0, 1, 1)
        self.controller_pulserev = QtWidgets.QLineEdit(self.centralwidget)
        self.controller_pulserev.setObjectName("controller_pulserev")
        self.gridLayout_2.addWidget(self.controller_pulserev, 1, 1, 1, 1)
        self.distance_per_rotation = QtWidgets.QLineEdit(self.centralwidget)
        self.distance_per_rotation.setObjectName("distance_per_rotation")
        self.gridLayout_2.addWidget(self.distance_per_rotation, 1, 2, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout_2)
        self.param_set = QtWidgets.QPushButton(self.centralwidget)
        self.param_set.setObjectName("param_set")
        self.horizontalLayout.addWidget(self.param_set)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_2.addWidget(self.label_8)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setKerning(True)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 2, 1, 1)
        self.main_speed = QtWidgets.QLineEdit(self.centralwidget)
        self.main_speed.setObjectName("main_speed")
        self.gridLayout.addWidget(self.main_speed, 1, 0, 1, 1)
        self.main_acceleration = QtWidgets.QLineEdit(self.centralwidget)
        self.main_acceleration.setObjectName("main_acceleration")
        self.gridLayout.addWidget(self.main_acceleration, 1, 1, 1, 1)
        self.main_distance = QtWidgets.QLineEdit(self.centralwidget)
        self.main_distance.setObjectName("main_distance")
        self.gridLayout.addWidget(self.main_distance, 1, 2, 1, 1)
        self.horizontalLayout_2.addLayout(self.gridLayout)
        self.main_start = QtWidgets.QPushButton(self.centralwidget)
        self.main_start.setObjectName("main_start")
        self.horizontalLayout_2.addWidget(self.main_start)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_2.addWidget(self.line)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_2.addWidget(self.label_7)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.jogdown = QtWidgets.QPushButton(self.centralwidget)
        self.jogdown.setObjectName("jogdown")
        self.gridLayout_3.addWidget(self.jogdown, 1, 2, 1, 1)
        self.jogup = QtWidgets.QPushButton(self.centralwidget)
        self.jogup.setObjectName("jogup")
        self.gridLayout_3.addWidget(self.jogup, 0, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 0, 0, 1, 1)
        self.jogspeed = QtWidgets.QLineEdit(self.centralwidget)
        self.jogspeed.setObjectName("jogspeed")
        self.gridLayout_3.addWidget(self.jogspeed, 1, 0, 1, 1)
        self.set_jog_speed = QtWidgets.QPushButton(self.centralwidget)
        self.set_jog_speed.setObjectName("set_jog_speed")
        self.gridLayout_3.addWidget(self.set_jog_speed, 1, 1, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout_3)
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout_2.addWidget(self.line_2)
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.verticalLayout_2.addWidget(self.label_9)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout_4.addWidget(self.label_4, 0, 0, 1, 1)
        self.csvstart = QtWidgets.QPushButton(self.centralwidget)
        self.csvstart.setObjectName("csvstart")
        self.gridLayout_4.addWidget(self.csvstart, 1, 1, 1, 1)
        self.csvaddress = QtWidgets.QLineEdit(self.centralwidget)
        self.csvaddress.setObjectName("csvaddress")
        self.gridLayout_4.addWidget(self.csvaddress, 1, 0, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout_4)
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.verticalLayout_2.addWidget(self.line_3)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.approxpos = QtWidgets.QLCDNumber(self.centralwidget)
        self.approxpos.setObjectName("approxpos")
        self.horizontalLayout_3.addWidget(self.approxpos)
        self.text_monitor = QtWidgets.QTextBrowser(self.centralwidget)
        self.text_monitor.setObjectName("text_monitor")
        self.horizontalLayout_3.addWidget(self.text_monitor)
        self.butStop = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.butStop.setFont(font)
        self.butStop.setObjectName("butStop")
        self.horizontalLayout_3.addWidget(self.butStop)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.gridLayout_5.addLayout(self.verticalLayout_2, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 566, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.param_set.clicked.connect(self.paramsetfx)
        #self.set_jog_speed.clicked.connect(self.jspeedsetfx)

        #Repeat send commands for jog buttons
        self.jogdown.setAutoRepeat(True)
        self.jogdown.setAutoRepeatDelay(0)
        self.jogdown.setAutoRepeatInterval(100)
        self.jogdown.pressed.connect(self.jogdownfx)

        self.jogup.setAutoRepeat(True)
        self.jogup.setAutoRepeatDelay(0)
        self.jogup.setAutoRepeatInterval(100)
        self.jogup.pressed.connect(self.jogupfx)
        self.tempstate = 0

        self.main_start.clicked.connect(self.mainstartfx)
        self.csvstart.clicked.connect(self.csvstartfx)



        #Relevant variables
    def paramsetfx(self):
        #Initiate parameters
        pulserev = float(self.controller_pulserev.text())
        distrot = float(self.distance_per_rotation.text())
        self.distperpulse = distrot/pulserev
        try:
            self.text_monitor.append('Initializing')
            self.coater = motioncontol(self.controller_COMport.text())
            time.sleep(2)
        except SerialException:
            self.text_monitor.append('reset')

    def mainstartfx(self):
        self.maind = float(self.main_distance.text())
        self.mainspd = float(self.main_speed.text())
        self.mainacc = float(self.main_acceleration.text())
        self.coater.setspeedacc(self.mainspd,self.mainacc,self.distperpulse)
        self.coater.moverelative(self.maind,self.distperpulse)

    def jogdownfx(self):
        if(self.jogdown.isDown()):
            if self.tempstate == 0:
                self.tempstate = 1
                self.jogspd = float(self.jogspeed.text())
                self.coater.setspeedacc(-1*self.jogspd,10000,self.distperpulse)
                self.coater.timejogmove(100)
            else:
                self.coater.timejogmove(100)
        else:
            self.tempstate = 0

    def jogupfx(self):
        if(self.jogup.isDown()):
            if self.tempstate == 0:
                self.tempstate = 1
                self.jogspd = float(self.jogspeed.text())
                self.coater.setspeedacc(self.jogspd,10000,self.distperpulse)
                self.coater.timejogmove(100)
            else:
                self.coater.timejogmove(100)
        else:
            self.tempstate = 0


    def csvstartfx(self):
        pass

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Chip"))
        self.label_5.setText(_translate("MainWindow", "Chip the Dip Coater and Fiber Puller Controller"))
        self.label_6.setText(_translate("MainWindow", "This is the GUI to control the DIY dipcoater/fiber puller named Chip set up in the lab."))
        self.label_13.setText(_translate("MainWindow", "COM port name"))
        self.label_10.setText(_translate("MainWindow", "Controller pulse/revolution"))
        self.label_11.setText(_translate("MainWindow", "Distance per rotation"))
        self.param_set.setText(_translate("MainWindow", "Set"))
        self.label_8.setText(_translate("MainWindow", "Speed + acceleration + distance"))
        self.label.setText(_translate("MainWindow", "Speed (mm/s)"))
        self.label_12.setText(_translate("MainWindow", "<html><head/><body><p>Acceleration (mm<span style=\" vertical-align:super;\">2</span>/s)</p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "Distance (mm)"))
        self.main_start.setText(_translate("MainWindow", "Start"))
        self.label_7.setText(_translate("MainWindow", "Jog"))
        self.jogdown.setText(_translate("MainWindow", "Down"))
        self.jogup.setText(_translate("MainWindow", "Up"))
        self.label_3.setText(_translate("MainWindow", "Speed (mm/s)"))
        self.set_jog_speed.setText(_translate("MainWindow", "Set Jog Speed"))
        self.label_9.setText(_translate("MainWindow", "csv speeds+times (not yet working)"))
        self.label_4.setText(_translate("MainWindow", "File Address"))
        self.csvstart.setText(_translate("MainWindow", "Start"))
        self.butStop.setText(_translate("MainWindow", "STOP"))

#Need to run in console
if __name__ == "__main__":
    import sys
    def run_app():

        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
        app = QtWidgets.QApplication(sys.argv)

        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        MainWindow.show()
        sys.exit(app.exec_())
    run_app()