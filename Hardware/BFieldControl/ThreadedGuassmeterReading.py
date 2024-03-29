# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:36:22 2023

@author: hamza
"""

import serial.tools.list_ports
import numpy as np
import msvcrt, os, keyboard
saveloc = 'testrec3.csv'
#ports = serial.tools.list_ports.comports()

#for port, desc, hwid in sorted(ports):
#       print("{}: {} [{}]".format(port, desc, hwid))
	
porttouse = 8
import serial, time, threading, struct, csv
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
		#Find out where in the output bytes 
		secondinfo = 47 #starting index of second bit
		signstart = 13 #bit number where sign info starts
		self.bitstart = secondinfo+signstart
		
	
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
					
					BSign = int(binarylist[self.bitstart])
					BDivide =binarylist[self.bitstart+1:self.bitstart+4].int
					print(BDivide)
					Bval = struct.unpack('>I',dat[-4:])[0]
					Bval = (-1)**BSign*Bval/(10**BDivide)
					print("Time: {t}, field: {B} Guass".format(t = t1-t0, B = Bval))
					#self.finalreading = dat
	def readB(self,serialin,binarylist):
		#Reads the B field from an appropriate 12 byte data chunk from the 0x03 command
		BSign = int(binarylist[self.bitstart])
		BDivide =binarylist[self.bitstart+1:self.bitstart+4].int
		Bval = struct.unpack('>I',serialin[-4:])[0]
		Bval = (-1)**BSign*Bval/(10**BDivide)
		return Bval

	def getDatum(self):
		checker = True
		toomany = 5
		i=0
		while(checker):		
			self.commandwrite(0x03)
			dat = self.ser.read(30)[:-1]
			if (len(dat)>1):
				binarylist = BitArray(dat)
				if (binarylist[0]==False): #Device throws 1 at first bit if data is to be ignored
					t1 = time.time()
					Bval = self.readB(dat, binarylist)
					checker=False
			i=i+1
			if(i>toomany):
				raise('Not reading properly')
				
		return [t1,Bval]
 
	def closeport(self):
		self.ser.close()






# Send command to device and save its return
meter = gaussmeter('COM8')
  
meter.getIdentification()

saveloc = input('Input file name to without a .csv, no number ending \n')
while os.path.isfile(os.path.abspath(saveloc+'.csv')):
	saveloc = saveloc+'1'
saveloc = saveloc + '.csv'
	


deltaT = input('Enter recording time in minutes')
deltaT = float(deltaT)*60

csv_file = open(saveloc, "w")

writer = csv.writer(csv_file, delimiter=',', lineterminator="\n")

t0 = time.time()
t1 = t0
print('starting')
runcheck = True
while ((t1-t0)<deltaT and runcheck):
	bs = meter.getDatum()
	#print("Time: {t}, field: {B} Guass".format(t = bs[0]-t0, B = bs[1]))
	writer.writerow([bs[0],bs[1]])
	t1=time.time()
	if keyboard.is_pressed('esc'):
		runcheck = False

#%%
meter.ser.close()


csv_file.close()