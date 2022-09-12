'''
This code is for quickly combining files for later analysis and then saving a video to make for easy playing
'''

import sys, os
import importlib
#need to install imageio and imagio-ffmpeg
import imageio
import numpy as np
#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved, use forward slashes
dataDR=r"F:\PDMSMigration\Tinydrops"


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
outFile="vid1.mp4"
infile='tinydrops.ome.tif'

#Import the tif files in a folder
#imageframes=ito.omestackimport(dataDR)
imageframes=ito.fullseqimport(os.path.join(dataDR,infile))

#Or just one file
#stackimport(dataDR+'/'+inFile)
#%%
imageframes1=imageframes[:,0]
imageframes2=imageframes[:,1]
'''
outFile1="Angle1.mp4"
outFile2="Angle2.mp4"
imageio.mimwrite(outFile1, imageframes1 ,quality=10, input_params=['-r','30'],  output_params=['-r', '30'])
imageio.mimwrite(outFile2, imageframes2 ,quality=10, input_params=['-r','30'],  output_params=['-r', '30'])
'''
#%%
#Write to a video using mimwrite
combo = np.concatenate((imageframes1, imageframes2), axis=1)
imageio.mimwrite("vid3.mp4", combo ,quality=5, input_params=['-r','5'],  output_params=['-r', '30'])
#%%
'''
#for single axis video
outFile1="evap.mp4"
imageio.mimwrite(outFile1, imageframes ,quality=10, input_params=['-r','10'],  output_params=['-r', '30'])
'''