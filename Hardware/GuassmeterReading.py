# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:36:22 2023

@author: hamza
"""

import serial.tools.list_ports
import numpy as np

#ports = serial.tools.list_ports.comports()

#for port, desc, hwid in sorted(ports):
#        print("{}: {} [{}]".format(port, desc, hwid))
		
porttouse = 4
import serial
import time
import struct
from bitstring import BitArray
def stb(x):
	#Shortening serial to bytes
	return serial.to_bytes(x)


class gaussmeter:
	def __init__(self,portval):
		self.ser=serial.Serial(
		        port = portval,
		        baudrate = 115200,
		        bytesize = 8,
		        stopbits = 1,
				timeout=.5,
		        parity = serial.PARITY_NONE)
		self.DEFAULT_SLEEP_TIME = 0.1
		#Define some common hex values
		self.ACKNOWLEDGE = 0x08
		self.RETURNACK = [0x08]*6
		self.TERMINATE= 0x07
		self.ID_METER_PROP = 0x01

	
	def commandwrite(self,incommand):
		'''
		Simple function to write a command given in the 0x01 type hex format to
		something that the device can understand
		Just need to input the first hex, this function will add the don't matter
		bytes
		'''
		#Add the don't matter bytes
		command = [incommand,0x03, 0x03, 0x03, 0x03, 0x03] 
		towrite = serial.to_bytes(command) #Convert to bytes
		self.ser.write(towrite) #Send to device
	def simplewrite(self,incommand):
		#Simply writing a command in hex to bytes
		towrite = serial.to_bytes(incommand)
		self.ser.write(towrite)
		
	def getIdentification(self):
		'''
		This function gets the identifying information for the guassmeter and 
		makes sure that it is running

		'''
		self.commandwrite(self.ID_METER_PROP) #Write the command requesting info
		meterdat = self.ser.read(21) #Read data, output from device should be 21 bytes
		info = meterdat[:-1] #Actual data is 20bytes
		infodec = info.decode('ascii')
		print(infodec)
		check = meterdat[-1] #Final byte tells us to continue or not
		readcheck = True
		if check == self.ACKNOWLEDGE:
			readcheck = True
			self.simplewrite(self.RETURNACK)
		if check == self.TERMINATE:
			readcheck = False
		while(readcheck):
			meterdat = self.ser.read(21) #Data output is 21 bits
			info = meterdat[:-1]
			infodec = info.decode('ascii')
			print(infodec)
			check = meterdat[-1]
			if check == self.TERMINATE:
				 readcheck = False
			else:
				self.simplewrite(self.RETURNACK)
		print('done reading')
	
	def getData(self,pointnum):
		t0=time.time()
		secondinfo = 47 #starting index of second bit
		signstart = 13 #bit number where sign info starts
		bitstart = secondinfo+signstart
		for i in range(pointnum):
			self.commandwrite(0x03)
			dat = self.ser.read(30)[:-1]
			#print(dat)
			if (len(dat) >1):
				binarylist = BitArray(dat)
				if (binarylist[0]==False): #Device throws 1 at first bit if data is to be ignored
					t1 = time.time()
					#print("{0:0.2E}".format(t1-t0))
					#print(dat)
					
					BSign = int(binarylist[bitstart])
					BDivide =binarylist[bitstart+1:bitstart+4].int
					print(BDivide)
					Bval = struct.unpack('>I',dat[-4:])[0]
					Bval = (-1)**BSign*Bval/(10**BDivide)
					print("Time: {t}, field: {B} Guass".format(t = t1-t0, B = Bval))
					t0=t1
					#self.finalreading = dat
		
 
	def closeport(self):
		self.ser.close()



# Send command to device and save its return
meter = gaussmeter('COM4')

meter.getIdentification()

meter.getData(30)

meter.ser.close()

#%%
