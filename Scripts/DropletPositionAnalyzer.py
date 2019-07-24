# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:54:38 2019

@author: Hamza
"""

import sys, os
import matplotlib.pyplot as plt
import numpy as np
import importlib
#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"E:\Newtips\1um_2"


os.chdir(CodeDR) #Set  current working direcotry to the code directory


sys.path.append('./Tools') #Add the tools to the system path so modules can be imported

#Import required modules
import DropletprofileFitter as df
importlib.reload(df)
import ImportTools as ito 
importlib.reload(ito)
import EdgeDetection as ede
importlib.reload(ede)
import Crosscorrelation as crco
importlib.reload(crco)


#Remove to avoid cluttering path
sys.path.remove('./Tools') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)
#%%
#Import the image
allimages=ito.stackimport(dataDR+r"\1um_2_MMStack_Default.ome.tif")
#%%
#Select the minimum (1s) and maximum (2s) crop locations
x1c=300
x2c=900
y1c=400
y2c=1000
croppoints=[x1c,x2c,y1c,y2c]

fig, ax = plt.subplots(nrows=2, ncols=2)
testimage1=allimages[0]
testimage2=allimages[-1]


croptest1=ede.cropper(testimage1,*croppoints)
croptest2=ede.cropper(testimage2,*croppoints)

ax[0,0].imshow(testimage1)
ax[0,1].imshow(testimage2)

ax[1,0].imshow(croptest1)
ax[1,1].imshow(croptest2)

#%%
#Crop all of the images and plot a cut at a y value
croppedimages=ede.cropper(allimages,*croppoints)

cutpixely=-50

a=croppedimages[0,cutpixely]
b=croppedimages[-1,cutpixely]
plt.plot(a)
plt.plot(b)
#%%
alldat=np.zeros([croppedimages.shape[0],croppedimages.shape[2]*2-1,2])
centerloc=np.zeros([croppedimages.shape[0],2])

a=croppedimages[0,-15]
for i in range(croppedimages.shape[0]):
    #The shift of the index 0 value represents the autocorrelation
    alldat[i]=crco.crosscorrelator(croppedimages[i,cutpixely],a)
    gparam, gfit = crco.centerfinder(alldat[i,:,0],alldat[i,:,1],20)
    centerloc[i]=[gparam[1],gfit[1]]

def xvtfinder(images,baseimage,cutloc,gausspts1):
    '''
    Takes a image sequence and the original image and returns series of shifts
    from the base image using cross correlation at the y pixel defined by cutloc
    gaussspts1 is the number of points to use in the gaussian fit on either side
    '''
    #Create empty array to store data
    centerloc=np.zeros([images.shape[0],2])
    #Perform cross correlation and use gaussian fit to find center position
    for i in range(croppedimages.shape[0]):
        alldat[i]=crco.crosscorrelator(images[i,cutloc],baseimage)
        gparam, gfit = crco.centerfinder(alldat[i,:,0],alldat[i,:,1],gausspts1)
        centerloc[i]=[gparam[1],gfit[1]]
    #Account for the 0 point
    centerloc = centerloc-[centerloc[0,0],0]
    return centerloc
    

    
np.save("datcorr",alldat)
np.save("centerloc",centerloc)

#%%
plt.plot(alldat[-20,:,0],alldat[-20,:,1],'.')

#%%
xvals=centerloc[:,0]-centerloc[0,0]
plt.plot(xvals,'.')
#%%

vel=np.gradient(xvals)
plt.plot(xvals,vel)
plt.xlabel('Position (pixels)')
plt.ylabel('droplet velocity')