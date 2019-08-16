'''
This code performs the edge location and cross correlation analysis across multiple images
'''

import sys, os, glob
import matplotlib.pyplot as plt
import numpy as np
import importlib
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import pickle
#%%
#import similaritymeasures

#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"E:\SpeedScan"


os.chdir(CodeDR) #Set  current working direcotry to the code directory


sys.path.append('./Tools') #Add the tools to the system path so modules can be imported

#Import required modules
import DropletprofileFitter as df
importlib.reload(df)
import Crosscorrelation as crco
importlib.reload(crco)
import ImportTools as ito 
importlib.reload(ito)
import EdgeDetection as ede
importlib.reload(ede)

#Remove to avoid cluttering path
sys.path.remove('./Tools') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)


#%%
#Check the extrema images and note the limits that make sense
noforce=ito.imread2(dataDR+'\\base.tif')
ex1=ito.imread2(dataDR+'\\extreme1.tif')
ex2=ito.imread2(dataDR+'\\extreme2.tif')

gs = gridspec.GridSpec(1, 3)

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax1.imshow(noforce)
ax2.imshow(ex1)
ax3.imshow(ex2)
#%%

#Specify parameters
#Cropping
#Select the minimum (1s) and maximum (2s) crop locations
#Needs to include the pipette ends
x1c=616
x2c=1412
y1c=500
y2c=855
croppoints=[x1c,x2c,y1c,y2c]

#Select crop region for fitting (just needs to be large enough so droplet end is the max)
yanlow=679
yanhigh=748
yanalysisc=[yanlow-y1c,yanhigh-y1c]

croppedbase=ede.cropper(noforce,*croppoints)
croppedex1=ede.cropper(ex1,*croppoints)
croppedex2=ede.cropper(ex2,*croppoints)

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(gs[0, 0])

ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax2.axhline(yanalysisc[0])
ax2.axhline(yanalysisc[1])
ax1.imshow(croppedbase)
ax2.imshow(croppedex1)
ax3.imshow(croppedex2)
x1c=616
x2c=1412
y1c=500
y2c=855
croppoints=[x1c,x2c,y1c,y2c]

#Cross correlation
cutpoint=50 # y pixel to use for cross correlation
guassfitl=20 # Number of data points to each side to use for guass fit

#Edge detection
imaparam=[-40,20,.05] #[threshval,obsSize,cannysigma]
fitfunc=df.pol2ndorder #function ie def(x,a,b) to fit to find properties
fitguess=[0,1,1]
clinyguess = 214 #Guess at the center line (helpful if parts of pipette are further than droplet)
pixrange=[60,25] #xy bounding box to use in fit
#Specify an image to use as a background (needs same dim as images being analysed)
#Or can set to False
background=False 


allimages=ito.folderstackimport(dataDR)
allimages=ede.cropper(allimages,*croppoints)
#%%

stackedges = ede.seriesedgedetect(allimages,background,*imaparam)
#Fit the edges and extract angles and positions
AnglevtArray, EndptvtArray = df.edgestoproperties(stackedges,pixrange,fitfunc,fitguess)

noforce=ito.imread2(dataDR+'\\base.tif')