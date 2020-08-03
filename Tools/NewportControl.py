'''
Code for communicating with motion controller, right now it simply opens using
the correct port. Can get port from the GUI or by cycling through
Included quick change enable disable etc with this as well but is ussually good
to use the SMC100 applet to do that.
Make sure to close the port or cannot rerun the program
'''
import serial
import time

class SMC100:
	def __init__(self,portval):
		self.ser=serial.Serial(
		        port = portval,
		        baudrate = 57600,
		        bytesize = 8,
		        stopbits = 1,
		        parity = 'N',
		        xonxoff = True,
		        timeout = 0.050)
	
	#General writes
	def writecommand(self,docuCommand):
		modstring=docuCommand +'\r\n'
		self.ser.write(modstring.encode())
		time.sleep(0.1) #need to sleep for a bit while command processed
		
	def getinfo(self,docuCommand):
		modstring=docuCommand +'\r\n'
		self.ser.write(modstring.encode())
		rawresult = self.ser.readline()
		result=rawresult[3:-2].decode('utf8')
		return 	result
	
	#Some useful ones for quick access
	def toready(self):
		self.writecommand('1MM1')
		time.sleep(1) #some extra time to get ready
		
	def torest(self):
		self.writecommand('1MM0')
		time.sleep(1) #some extra time to sleep
	
	def setspeed(self,speed):
		'''
		Give speed in mm/s
		'''
		self.writecommand('1VA'+str(speed))
	
	def goto(self,position):
		'''
		Goes to a specific position, limit to <15
		'''
		if position<15:
			self.writecommand('1PA'+str(position))
		else:
			raise ValueError('Position too high')
	
	def closeport(self):
		self.ser.close()

'''


Example Usage

testSMC= SMC100('COM4')
testSMC.toready()
testSMC.setspeed('0.200')
testSMC.goto(8)
for i in range(15):
	toops = testSMC.getinfo('1TP')
	print(toops)
	time.sleep(0.1)
testSMC.torest()
testSMC.closeport()

'''