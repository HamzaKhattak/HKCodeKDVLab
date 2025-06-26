# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 17:50:26 2025

@author: hamza
"""

import cv2
import time
import numpy as np
import serial
import matplotlib.pyplot as plt

#Need the RTS DTR stuff off so the ESP doesn't reset after closing serial
ser = serial.Serial('COM4',115200,dsrdtr=None)
ser.setRTS(False)
ser.setDTR(False)

cam = cv2.VideoCapture(1)

if not cam.isOpened():
    print("Cannot open camera")
    exit()
nframes = 0
ts = 0
i=0

approxfps=40

acceleration = 20000



step1range=10000

step1speed = 10000

step2range = 3000
step2speed = 1000

step3range = 200
step3speed = 50

def emptydatarraygen(distance,speed,fps):
    tapprox = distance/speed
    numFramesapprox = tapprox*fps
    return np.zeros([int(tapprox*fps*10),2])


def compute_laplacian(image):
    ''' image should be grayscale '''
    laplacian = cv2.Laplacian(image, cv2.CV_64F)  # Apply Laplacian filter
    return np.var(laplacian)  # Compute variance of Laplacian


def compute_sobel_variance(image):
    ''' image should be grayscale '''
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel X gradient
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel Y gradient
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)  # Compute gradient magnitude
    variance = np.var(image)  # Compute variance of pixel intensities
    return np.mean(sobel_magnitude) + variance  # Combine Sobel and variance


def __draw_label(img, text, pos, bg_color):
   font_face = cv2.FONT_HERSHEY_SIMPLEX
   scale = 2
   color = (0, 0, 0)
   thickness = cv2.FILLED
   margin = 2
   txt_size = cv2.getTextSize(text, font_face, scale, thickness)

   end_x = pos[0] + txt_size[0][0] + margin
   end_y = pos[1] - txt_size[0][1] - margin

   cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
   cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

def writecommand(str):
    ser.write(str.encode(encoding="utf-8") )

def read():
    return ser.readline().rstrip().decode("utf-8")

def move(loc):
    command = 'm' + str(loc) + '\n'
    writecommand(command)

def setspeed(speed,acceleration):
    command = 's' + str(speed) + ',' + str(acceleration) + '\n'
    writecommand(command)
#Initial loop to get focus close manually
def imageonly():
    while True:
        t1=time.time()
        
        ret, frame = cam.read()
        writecommand('l')
        location = int(read()) #kinda inefficient doesn't slow things down

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        focusval = compute_laplacian(gray)
        # Display the captured frame
        cv2.imshow('Camera', frame)

        keyPress = cv2.waitKey(1)
        if keyPress == ord('q'):
            break


def captureloop(j,arr):
    cont=True
    while(cont):
        j=j+1
        ret, frame = cam.read()
        writecommand('l')
        location = int(read()) #kinda inefficient doesn't slow things down
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        focusval = compute_laplacian(gray)
        arr[j] = [location,focusval]
        cv2.imshow('Camera', frame)
        cv2.waitKey(1)
        if j%20==0:
            writecommand('g')
            check = read()
            if check=='d':
                cont = False
    return j

imageonly()



#Get the first frame
ret, frame = cam.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
focusval = compute_laplacian(gray)
writecommand('l')
location = int(read()) #kinda inefficient doesn't slow things down

step1limits = [location-int(step1range/2),location+int(step1range/2)]
firstarray = emptydatarraygen(step1limits[1]-step1limits[0],step1speed,approxfps)
firstarray[i] = [location,focusval]



#Move to start of range and get first array
setspeed(step1speed,acceleration)
move(step1limits[0])
i = captureloop(i,firstarray)

move(step1limits[1])
i = captureloop(i,firstarray)


maxloc = np.argmax(firstarray[:,1])

step2limits = [firstarray[maxloc,0]-int(step2range/2),firstarray[maxloc,0] + int(step2range/2)]
secondarray = emptydatarraygen(step2limits[1]-step2limits[0],step2speed,approxfps)
setspeed(step2speed,acceleration)



i=0
move(step2limits[0])
i = captureloop(i,secondarray)

move(step2limits[1])
i = captureloop(i,secondarray)

maxloc = np.argmax(secondarray[:,1])


move(int(secondarray[maxloc,0]))


imageonly()

ser.close()
cam.release()
cv2.destroyAllWindows()


plt.plot(firstarray[:i,0],firstarray[:i,1],'.')
plt.plot(secondarray[:i,0],secondarray[:i,1],'.')
plt.show()