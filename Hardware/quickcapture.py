'''
This code gets a quick sequence of images
'''

import sys, os, glob, pickle, re
import numpy as np
import importlib
import time
#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"F:\PDMSMigration\Tinydrops"

²
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

numFrames=300
framespersec=.005

secperframe = 1/framespersec
#Open the camera and controller
cam=cseq.BCamCap(2,secperframe)

cam.grabFastSequence(int(numFrames),"dualdroplets41")

