# -*- coding: utf-8 -*-
"""
Created on Mon May 27 13:56:55 2019

@author: Hamza
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage import feature
from skimage.filters import threshold_otsu
from skimage.filters import threshold_minimum
import skimage.morphology as morph

from scipy.optimize import curve_fit
from scipy.misc import derivative
#%%

background=ndi.imread('background.png',flatten=True)
im = ndi.imread('Images/moveseq103.png',flatten=True)

imsub=background-im

threshimage=imsub>10
plt.imshow(threshimage,cmap=plt.cm.gray)


oRem=morph.remove_small_objects(threshimage,100)
hRem=morph.remove_small_holes(oRem,100)
plt.imshow(hRem,cmap=plt.cm.gray)

edgedetect=feature.canny(hRem, sigma=.05)
plt.imshow(edgedetect, cmap=plt.cm.gray)
locs=np.argwhere(edgedetect)

#%%

plt.plot(locs[:,1],locs[:,0],'.',markersize=1)
plt.xlim(100,600)
plt.axes().set_aspect('equal')
plt.tight_layout()

#%%
yminval=460
ymaxval=690

interval=np.logical_and(locs[:,0]>yminval,locs[:,0]<ymaxval)
impDat=locs[interval]

plt.plot(impDat[:,1],impDat[:,0],'.',markersize=2)

splitline=np.mean(impDat[:,1][impDat[:,0]>ymaxval-10])

plt.axvline(splitline)
leadedge=np.max(impDat[:,1])
trailedge=np.min(impDat[:,1])
plt.axvline(leadedge)
plt.axvline(trailedge)
#%%
leftdat=impDat[impDat[:,1]<splitline]
plt.plot(leftdat[:,1],leftdat[:,0],'.',markersize=2)

#%%
def circle(x,a,b,r):
    return np.sqrt(r**2-(x-a)**2)+b


minLoc=np.argmin(leftdat[:,1])
minx=leftdat[minLoc,1]
miny=leftdat[minLoc,0]
pixelbuff=30
conds=np.logical_and(leftdat[:,1]<minx+pixelbuff,leftdat[:,0]>miny)
trimDat=leftdat[conds]-[miny,minx]
plt.plot(trimDat[:,1],trimDat[:,0])
#%%
minLoc2=np.argmin(trimDat)
sigma = np.ones(len(trimDat[:,0]))
sigma[minLoc2] = 1
popt, pcov = curve_fit(circle, trimDat[:,1], trimDat[:,0],p0=[2,10,30], sigma=sigma)

maxtrimVal=np.max(trimDat)
x=np.linspace(0,maxtrimVal+50,100)
plt.plot(x,circle(x,*popt))
plt.plot(leftdat[:,1]-minx,leftdat[:,0]-miny)

def paramcirc(x):
    return circle(x,*popt)
mcirc=derivative(paramcirc,0)
thet=np.arctan(mcirc)
def slopeptline(x,m,x0,y0):
    return m*(x-x0)+y0

x2=np.linspace(-10,15,100)
plt.plot(x2,slopeptline(x2,mcirc,0,0))
plt.ylim(-50,50)
plt.axes().set_aspect('equal')

print(thet*180/np.pi)

#%%

plt.plot(leftdat[:,1],leftdat[:,0],'.',markersize=2)
plt.plot(leftdat[minLoc,1],leftdat[minLoc,0],'o',markersize=3)

#%%
def endfind(main,back,imthresh,obsize,yminval,ymaxval):
    '''
    This function will return the start and end of a droplet
    main is the string indicating the location of the main file and back
    is the string for the background
    imthresh is the threshold value applied
    #ymin and ymax are used to select the portion of the image to include
    '''
    #Import the images
    background=ndi.imread(back,flatten=True)
    im = ndi.imread(main,flatten=True)
    #Subtract the background and apply the threshold
    imsub=background-im
    threshimage=imsub>imthresh
    
    #Remove dust specs
    oRem=morph.remove_small_objects(threshimage,obsize)
    hRem=morph.remove_small_holes(oRem,obsize)
    
    #Find the edges
    edgedetect=feature.canny(hRem, sigma=.05)
    #Convert to xy values
    locs=np.argwhere(edgedetect)
    
    #Select the correct region
    interval=np.logical_and(locs[:,0]>yminval,locs[:,0]<ymaxval)
    impDat=locs[interval]

    
    leadedge=np.max(impDat[:,1])
    trailedge=np.min(impDat[:,1])
    return [trailedge,leadedge]

#%%
import glob
imfilenames=glob.glob("Images/*.png")
numFiles=len(imfilenames)
#%%    
timeLocArray=np.zeros([numFiles,2])
for i in range(numFiles):
    timeLocArray[i]=endfind(imfilenames[i],"background.png",10,100,460,690)
    


#%%
im = ndi.imread('Translate1ums5xob.tif')
#%%
plt.subplots(1,2, figsize=(6,4))
rejig=(-timeLocArray[4:]+1000)
velt=-np.gradient(rejig[:,0])
vell=-np.gradient(rejig[:,1])

#%%
ax1=plt.subplot(2, 1, 1)
ax1.plot(rejig[:,0],label='trailing')
ax1.plot(rejig[:,1],label='leading')
ax1.legend()
ax1.set_xlabel("timestep")
ax1.set_ylabel("position (pixels)")

ax2=plt.subplot(2, 1, 2)
ax2.plot(rejig[:,0],velt,'.',label='trailing')
ax2.plot(rejig[:,1],vell,'.',label='leading')
ax2.legend()
ax2.set_ylabel("speed (au)")
ax2.set_xlabel("position (pixels)")
plt.tight_layout()