'''
This code performs the edge location and cross correlation analysis across multiple images
'''

import sys, os, glob, pickle, re
import matplotlib.pyplot as plt
import numpy as np
import importlib
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter
from matplotlib_scalebar.scalebar import ScaleBar
#import similaritymeasures
import time
#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"F:\Fiber"


os.chdir(CodeDR) #Set  current working direcotry to the code directory


sys.path.append('./Tools') #Add the tools to the system path so modules can be imported

#Import required modules
import CameraSequencer as cseq
importlib.reload(cseq)
import NewportControl as nwpt
importlib.reload(nwpt)
#Remove to avoid cluttering path
sys.path.remove('./Tools') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)
#%%
movespeed=0.02
movedistance=1
numFrames=200
framespersec=movedistance/movespeed/numFrames

cont=nwpt.SMC100('COM4')
cont.toready()
position=float(cont.getpos())

movespeed=0.05
movedistance=-1
numFrames=100

secperframe = np.abs(2*movedistance/movespeed/numFrames)
#Open the camera and controller
cam=cseq.BCamCap(2,secperframe)
#Set the speed
cont.setspeed(movespeed)
#Move to the end points and capture frames
#Will have extra header for second go around since no multithreading yet
cont.goto(position+movedistance)
cam.grabSequence(int(np.floor(numFrames/2)),"pipette2")
time.sleep(4)
cont.stop()
time.sleep(2)
cont.goto(position)
cam.grabSequence(int(np.ceil(numFrames/2)),"pipette2")


cont.torest()
cont.closeport()
