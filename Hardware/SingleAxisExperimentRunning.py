'''
This code performs the edge location and cross correlation analysis across multiple images
'''

import sys, os, glob, pickle, re
import numpy as np
import importlib
import time
#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"F:\Finger\Experiments\20persample"


os.chdir(CodeDR) #Set  current working direcotry to the code directory


sys.path.append('./Tools') #Add the tools to the system path so modules can be imported

#Import required modules
import CameraSequencer1axis as cseq
importlib.reload(cseq)
import NewportControl as nwpt
importlib.reload(nwpt)
import ImportTools as ito
importlib.reload(ito)
#Remove to avoid cluttering path
sys.path.remove('./Tools') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)

#%%
'''
#CSV with run parameters, columns as noted below:
# Speed_ums, Point_1_mm, Point_2_mm, Distance_mm, Number_of_frames, Secperframe, Distance_per_frame, Repeats
#Best to name files with date and sample ID
#For multiple runs at same speed simply have the run multiple times in the file
#Experiments run in the order they are in the csvW
The number of frames refers to each run which goes back and forth once
'''



runparams = np.genfromtxt('runinfo.csv', dtype=float, delimiter=',', names=True) 
speedarray = runparams[r"Speed_ums"]/1000 #Speed is inputted into the device in mm/s
limit1Array = runparams[r"Point_1_mm"] #Point 1 and Point 2 are locations in mm
limit2Array = runparams[r"Point_2_mm"]
numFrameArray = runparams[r"Number_of_frames"]
repeatnum = runparams[r"Repeats"]


#Run the experiments, for now need to have it start and end at same points
#Should be able to implement multi threading and a bit more complicated wait cycles later
#Open the controller
cont=nwpt.SMC100('COM1')
cont.toready()
try:
	for i in np.arange(len(speedarray)):
		#Repeat for the number of repeats required
		for j in np.arange(repeatnum[i]):
			#Create folder and file saving name
			foldname = ito.spfoldercreate(speedarray[i])
			folddir=os.path.join(dataDR,foldname)
			os.chdir(folddir)
			filesavename= foldname + 'run'
			print(filesavename+'inst'+str(i)+'-'+str(j)+'Started')
			#Find the seconds per frame
			distance = np.abs(limit1Array[i]-limit2Array[i])
			secperframe = 2*distance/speedarray[i]/numFrameArray[i]
			#Open the camera and controller
			cam=cseq.BCamCap('40047306', secperframe)
			#Set the speed
			cont.setspeed(speedarray[i])
			#Move to the end points and capture frames
			#Will have extra header for second go around since no multithreading yet
			cont.goto(limit2Array[i])
			cam.grabSequence(int(np.floor(numFrameArray[i]/2)),filesavename)
			time.sleep(4)
			cont.stop()
			time.sleep(2)
			cont.goto(limit1Array[i])
			cam.grabSequence(int(np.ceil(numFrameArray[i]/2)),filesavename)
			os.chdir(dataDR)
			time.sleep(20) #Sleep for a bit before restarting
			cont.stop()
			time.sleep(2)
			print('Run Ended')
finally:
	cont.torest()
	cont.closeport()