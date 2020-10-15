



from PyQt5 import QtCore, QtGui, QtWidgets

import time
import serial
from serial import SerialException

import Dipfrontend as fedip
import Dipbackend as bedip


class FullGUI(fedip.Ui_MainWindow):

  #def __init__(self,object):
  #  super().__init__(object)

    def setupUiMOD(self,MainWindow):
        '''
        Set up the GUI from the front end and connect to functions
        '''
        self.setupUi(MainWindow)
        self.param_set.clicked.connect(self.paramsetfx)

        #Repeat send commands for jog buttons
        self.jogdown.setAutoRepeat(True)
        self.jogdown.setAutoRepeatDelay(0)
        self.jogdown.setAutoRepeatInterval(100)
        self.jogdown.clicked.connect(self.jogdownfx)

        self.jogup.setAutoRepeat(True)
        self.jogup.setAutoRepeatDelay(0)
        self.jogup.setAutoRepeatInterval(100)
        self.jogup.clicked.connect(self.jogupfx)
        self._tempstate = 0

        self.main_start.clicked.connect(self.mainstartfx)
        self.csvstart.clicked.connect(self.csvstartfx)



        #Relevant variables
    def paramsetfx(self):
        #Initiate parameters
        pulserev = float(self.controller_pulserev.text())
        distrot = float(self.distance_per_rotation.text())
        self.distperpulse = distrot/pulserev
        self.text_monitor.append('Setting')
        try:
            self.coater = bedip.motioncontol(self.controller_COMport.text())
            time.sleep(2)
        except SerialException:
            self.text_monitor.append('Warning: serial exception')

    def mainstartfx(self):
        self.maind = float(self.main_distance.text())
        self.mainspd = float(self.main_speed.text())
        self.mainacc = float(self.main_acceleration.text())
        self.coater.setspeedacc(self.mainspd,self.mainacc,self.distperpulse)
        time.sleep(.1) #Give time to process
        self.coater.moverelative(self.maind,self.distperpulse)

    def jogdownfx(self):
        if self.jogdown.isDown():
            if self._tempstate == 0:
                self._tempstate = 1
                #What happens on the initial click
                self.jogspd = float(self.jogspeed.text())
                self.jogtstep = float(self.jogtimestep.text())
                self.coater.setspeedacc(-1*self.jogspd,1000,self.distperpulse)
                time.sleep(.1) #Give time to process
            else:
                #What repeats
                self.coater.timejogmove(self.jogtstep)
        elif self._tempstate == 1:
            self._tempstate = 0
            #If needed what happens on release
        else:
            #If something needed for super short click
            pass


    def jogupfx(self):
        if self.jogup.isDown():
            if self._tempstate == 0:
                self._tempstate = 1
                #What happens on the initial click
                self.jogspd = float(self.jogspeed.text())
                self.jogtstep = float(self.jogtimestep.text())
                self.coater.setspeedacc(self.jogspd,1000,self.distperpulse)
                time.sleep(.1) #Give time to process
            else:
                #What repeats
                self.coater.timejogmove(self.jogtstep)
        elif self._tempstate == 1:
            self._tempstate = 0
            #If needed what happens on release
        else:
            #If something needed for super short click
            pass

    def csvstartfx(self):
        pass

#Need to run in console
if __name__ == "__main__":
    import sys
    def run_app():
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        ui = FullGUI()
        ui.setupUiMOD(MainWindow)
        MainWindow.show()
        sys.exit(app.exec_())
    run_app()