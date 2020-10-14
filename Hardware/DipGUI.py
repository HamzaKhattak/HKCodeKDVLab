



from PyQt5 import QtCore, QtGui, QtWidgets

import serial
import time
from serial import SerialException

import Dipfrontend as dipfe

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

class FullGUI(dipfe.Ui_MainWindow):

  #def __init__(self,object):
  #  super().__init__(object)

    def setupUiMOD(self,MainWindow):
        '''
        Set up the GUI from the front end and connect to functions
        '''
        self.setupUi(MainWindow)
        self.param_set.clicked.connect(self.paramsetfx)
        #self.set_jog_speed.clicked.connect(self.jspeedsetfx)

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
            self.coater = motioncontol(self.controller_COMport.text())
            time.sleep(2)
        except SerialException:
            self.text_monitor.append('Warning: serial exception')

    def mainstartfx(self):
        self.maind = float(self.main_distance.text())
        self.mainspd = float(self.main_speed.text())
        self.mainacc = float(self.main_acceleration.text())
        self.coater.setspeedacc(self.mainspd,self.mainacc,self.distperpulse)
        self.coater.moverelative(self.maind,self.distperpulse)

    def jogdownfx(self):
        if self.jogdown.isDown():
            if self._tempstate == 0:
                self._tempstate = 1
                #What happens on the initial click
                self.jogspd = float(self.jogspeed.text())
                self.coater.setspeedacc(-1*self.jogspd,1000,self.distperpulse)
                time.sleep(.1) #Give time to process
            else:
                #What repeats
                self.coater.timejogmove(100)
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
                self.coater.setspeedacc(self.jogspd,1000,self.distperpulse)
                time.sleep(.1) #Give time to process
            else:
                #What repeats
                print('hold')
                self.coater.timejogmove(100)
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