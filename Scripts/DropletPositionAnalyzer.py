# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:54:38 2019

@author: Hamza
"""

import sys, os, glob
import matplotlib.pyplot as plt
import numpy as np
import importlib
#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"E:\Newtips\SpeedAnalysis"


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
imagestack=ito.stackimport(dataDR+r"\1ums.tif")
#%%
#Select the minimum (1s) and maximum (2s) crop locations
x1c=300
x2c=900
y1c=400
y2c=1000
croppoints=[x1c,x2c,y1c,y2c]

fig, ax = plt.subplots(nrows=2, ncols=2)
testimage1=imagestack[0]
testimage2=imagestack[-1]


croptest1=ede.cropper(testimage1,*croppoints)
croptest2=ede.cropper(testimage2,*croppoints)

ax[0,0].imshow(testimage1)
ax[0,1].imshow(testimage2)

ax[1,0].imshow(croptest1)
ax[1,1].imshow(croptest2)

#%%
#Crop all of the images and plot a cut at a y value to test correlation shift
croppedimages=ede.cropper(imagestack,*croppoints)

cutpixely=-50

a=croppedimages[0,cutpixely]
b=croppedimages[-1,cutpixely]
plt.plot(a)
plt.plot(b)
#%%
#Test a couple of cut points
cutpoint=20
cutpoint2=30
a=croppedimages[0,50]
xvals,allcorr=crco.xvtfinder(croppedimages,a,cutpoint,20)
xvals2,allcorr2=crco.xvtfinder(croppedimages,a,cutpoint2,20)
plt.errorbar(np.arange(len(xvals)),xvals[:,0],yerr=xvals[:,1])   
plt.errorbar(np.arange(len(xvals)),xvals2[:,0],yerr=xvals2[:,1])    
#%%
vel=np.gradient(xvals[:,0])
plt.plot(xvals[:,0],vel)
plt.xlabel('Position (pixels)')
plt.ylabel('droplet velocity')

#%%
'''
Running multiple images
'''

filenames=glob.glob("*.tif")
PosvtArray=[None]*len(filenames)
for i in range(len(filenames)): 
    imagestack=ito.stackimport(dataDR + '\\' + filenames[i])
    croppedimages=ede.cropper(imagestack,*croppoints)
    xvals,allcorr=crco.xvtfinder(croppedimages,a,cutpoint,20)
    PosvtArray[i]=xvals[:,0]
    
#%%
plt.plot(PosvtArray[0],label='1ums')
plt.plot(PosvtArray[1],label='5ums')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel(r'Position $\alpha F$')
plt.tight_layout()
plt.savefig('Posvtime.png',dpi=300)