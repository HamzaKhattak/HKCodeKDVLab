'''
This code is for quickly combining files for later analysis and then saving a video to make for easy playing
'''

import sys, os
import importlib
#need to install imageio and imagio-ffmpeg
import imageio

#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved, use forward slashes
dataDR=r"E:\DualAngles\testing\run1_5"


os.chdir(CodeDR) #Set  current working direcotry to the code directory


sys.path.append('./Tools') #Add the tools to the system path so modules can be imported

#Import required modules
import ImportTools as ito 
importlib.reload(ito)

#Remove to avoid cluttering path
sys.path.remove('./Tools') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)

#%%
outFile="ionicrun.mp4"


#Import the tif files in a folder
imageframes=ito.omestackimport(dataDR)


#Or just one file
#stackimport(dataDR+'/'+inFile)
#%%
imageframes1=imageframes[:,0]
imageframes2=imageframes[:,1]

outFile1="Angle1.mp4"
outFile2="Angle2.mp4"
imageio.mimwrite(outFile1, imageframes1 ,quality=10, input_params=['-r','30'],  output_params=['-r', '30'])
imageio.mimwrite(outFile2, imageframes2 ,quality=10, input_params=['-r','30'],  output_params=['-r', '30'])

#%%
#Write to a video using mimwrite
imageio.mimwrite(outFile, imageframes ,quality=5, input_params=['-r','30'],  output_params=['-r', '30'])
