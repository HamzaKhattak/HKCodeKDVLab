# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 10:57:53 2024

@author: Dell
"""

import serial
import time
from serial.tools import list_ports
# Open serial port
ports = serial.tools.list_ports.comports()

for port, desc, hwid in sorted(ports):
        print("{}: {} [{}]".format(port, desc, hwid))
#%%
ser = serial.Serial('COM5', 9600,timeout=2)  # Adjust 'COM1' and baud rate as needed
time.sleep(1)

#First input denotes action. Second input must be 5 char. and decides which motors are affected
#Third input is integer command (ie for position), fourth input float input (ie for speed and acc)
ser.write(b"SS,11111,0,20000\n\r")
time.sleep(.1)
line = ser.readline()  
line2 = ser.readline()
print(line2)

# Close the serial port
ser.close()

