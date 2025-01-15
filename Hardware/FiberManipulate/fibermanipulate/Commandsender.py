# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 10:57:53 2024

@author: Dell
"""

import serial
import time
# Open serial port
ports = serial.tools.list_ports.comports()

for port, desc, hwid in sorted(ports):
        print("{}: {} [{}]".format(port, desc, hwid))

ser = serial.Serial('COM5', 9600,timeout=2)  # Adjust 'COM1' and baud rate as needed

time.sleep(1)
#%%
def read_from_teensy(ser):
	"""Read data from Teensy and print it."""
	data = ''
	if ser.in_waiting > 0:
		data = ser.read_until(b'\n').decode('utf-8').strip()
	return data
		
		
def write_to_teensy(ser, message):
    """Send data to Teensy."""
    ser.write(message.encode('utf-8') + b"\n")  # Send message with newline
	

def writeandread(ser,message):
	write_to_teensy(ser, message)
	time.sleep(.05)
	result = read_from_teensy(ser)
	time.sleep(.05)
	return result
	
def enablemotors():
	return writeandread(ser, 'EN,1,1,1')
def disablemotors():
	return writeandread(ser, 'EF,1,1,1')

def tensile(distance,stretch):
	if stretch==True:
		d1=distance
		d2=-distance
	else:
		d1=-distance
		d2=distance
	
	command1 = 'TR,00100,'+str(d1)+',1'
	command2 = 'TR,01000,'+str(d2)+',1'
	
	writeandread(ser,command1)
	writeandread(ser, command2)
	print(writeandread(ser, 'MS,1,1,1'))
	
	
def rotatetogether(speed,distance):
	speedcommand = 'SS,00011,1,'+str(speed)
	writeandread(ser, speedcommand)
	distancecommand = 'TR,00011,'+str(distance)+',1'
	writeandread(ser,distancecommand)
	print(writeandread(ser, 'MR,1,1,1'))
	
def dipmove(speed,distance):
	speedcommand = 'SS,10000,1,'+str(speed)
	writeandread(ser, speedcommand)
	distancecommand = 'TR,10000,'+str(distance)+',1'
	writeandread(ser,distancecommand)
	print(writeandread(ser, 'MD,1,1,1'))


#%%
#First input denotes action. Second input must be 5 char. and decides which motors are affected
#Third input is integer command (ie for position), fourth input float input (ie for speed and acc)
#Set speeds
ser.flush()
#%%
#25000 steps per rotation for rotators
#5mm pitch and 10000 steps per rotation for linear
def degtostep(degrees,microstepping):
	numsteps = (degrees/360)*microstepping
	return int(numsteps)

def disttostep(distance,pitch,microstepping):
	numstep = (distance/pitch)*microstepping
	return int(numstep)

print(disttostep(20,5,10000))

#%%
print(enablemotors())



#%%
rotatetogether(5000,6250)

#%%
tensile(20000,False)

#%%
tensile(3000,False)
#%%
dipmove(5000,-40000)
#%%
print(disablemotors())
#%%
# Close the serial port
ser.close()

