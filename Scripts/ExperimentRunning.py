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

#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"E:\AutoCaptureTest"


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
movespeed=0.01
movedistance=1
numFrames=200
framespersec=movedistance/movespeed/numFrames

cam=cseq.BCamCap(2,framespersec)
cont=nwpt.SMC100('COM4')
cont.setspeed(movespeed)

cont.goto(6)
cam.grabSequence(numFrames,'BobTest18')
cont.closeport()
