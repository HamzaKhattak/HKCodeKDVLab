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
CodeDR="F:\TrentDrive\Research\KDVLabCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR="F:\TrentDrive\Research\Droplet forces film gradients\SlideData2"


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
allimages=ito.stackimport(dataDR+"\Translate1ums5xob.tif")
#Select the minimum (1s) and maximum (2s) crop locations
x1c=9
x2c=750
y1c=715
y2c=898
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
a=croppedimages[0,-10]
b=croppedimages[-1,-10]
plt.plot(a)
plt.plot(b)
#%%
alldat=np.zeros([croppedimages.shape[0],croppedimages.shape[2]*2-1,2])
centerloc=np.zeros([croppedimages.shape[0],2])

a=croppedimages[0,-15]
for i in range(croppedimages.shape[0]):
    alldat[i]=crco.crosscorrelator(croppedimages[i,-15],a)
    gparam, gfit = crco.centerfinder(alldat[i,:,0],alldat[i,:,1],20)
    centerloc[i]=[gparam[1],gfit[1]]
    
    
np.save("datcorr",alldat)
np.save("centerloc",centerloc)

#%%
plt.plot(alldatx[-20],alldaty[-20],'.')


#%%
corrresultstest=crco.ccorrf(croppedimages[0,-10],croppedimages[-20,-10])

gparam, gfit = crco.centerfinder(corrresultstest[:,0],corrresultstest[:,1],20)

gaussfunctest=crco.gaussfunc(corrresultstest[:,0],*gparam)

plt.plot(corrresultstest[:,0],corrresultstest[:,1],'.')

plt.plot(corrresultstest[:,0],gaussfunctest)

#%%
xvals=centerloc[:,0]
vel=np.gradient(xvals)
plt.plot(xvals,vel,'.',markersize=1)
plt.xlabel('Position (pixels)')
plt.ylabel('droplet velocity')