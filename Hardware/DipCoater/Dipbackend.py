import serial
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