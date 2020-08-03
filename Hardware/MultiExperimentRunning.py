'''
This code performs the edge location and cross correlation analysis across multiple images
'''

import sys, os, glob, pickle, re
import matplotlib.pyplot as plt
import numpy as np
import importlib
import time
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
import ImportTools as imto
importlib.reload(imto)
#Remove to avoid cluttering path
sys.path.remove('./Tools') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)
#%%
'''
#CSV with run parameters, columns as noted below:
# Speed_ums, Point_1, Point_2, Distance, Number_of_frames, seconds/frame, distance per frame, Repeats
#Best to name files with date and sample ID
#For multiple runs at same speed simply have the run multiple times in the file
#Experiments run in the order they are in the csv
The number of frames refers to each run which goes back and forth once
'''
runparams = np.genfromtxt('Feb14-PDMSIonic150.csv', dtype=float, delimiter=',', names=True) 
speedarray = runparams[r"Speed_ums"]/1000 #Speed is inputted into the device in mm/s
limitArray = runparams[r"Point_1",r"Point_2"] #Point 1 and Point 2 are locations in mm
numFrameArray = runparams[r"Number_of_frames"]
repeatnum = runparams[r"Repeats"]



def foldercreate(runspeed):
	#Creates a unique folder in the working directory for a given runspeed
	#Avoid periods and special characters in numbers so 0.1um/s becomes 0p1ums etc
	#When reading simply replace the p with a "."
	prefix=str{runspeed*1000}
	prefix=prefix.replace(".","p")
	prefix = prefix + "ums"
	i = 0
	while os.path.exists(prefix+str(i)):
		i+=1
	createdfolder = prefix+str(i)
	os.mkdir(createdfolder)
	return createdfolder

#Run the experiments, for now need to have it start and end at same points
#Should be able to implement multi threading and a bit more complicated wait cycles later
for i in np.arange(len(speedarray)):
	#Repeat for the number of repeats required
	for j in np.arange(repeatnum[i])
		#Create folder and file saving name
		foldname = foldercreate(speedarray[i])
		filesavename= foldname + 'run'
		#Find the seconds per frame
		distance = np.abs(limitArray[i,1]-limitArray[i,0])
		secperframe = distance/movespeed/numframes/2
		#Open the camera and controller
		cam=cseq.BCamCap(2,secperframe)
		cont=nwpt.SMC100('COM4')
		#Set the speed
		cont.setspeed(movespeed)
		#Move to the end points and capture frames
		#Will have extra header for second go around since no multithreading yet
		cont.goto(limitArray[i,1])
		cam.grabSequence(np.floor(numFrameArray[i]/2),filesavename)
		cont.goto(limitArray[i,0])
		cam.grabSequence(np.ceil(numFrameArray[i]/2),filesavename)
		cont.closeport()
		time.sleep(5) #Sleep for a bit before restarting