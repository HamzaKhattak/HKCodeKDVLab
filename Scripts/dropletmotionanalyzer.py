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
from skimage import io
import scipy.ndimage.morphology as morph2
#%%
'''
Read in sequence and background if necessary 
'''
imsequence=io.imread('Translate1ums5xob.tif')
#background=io.imread('imagename.tif')
background=np.zeros(imsequence[0].shape)
#%%
#display sample for cropping, movement is expected in the x axis
plt.figure()
plt.imshow(imsequence[0])
plt.show()
plt.figure()
plt.imshow(imsequence[-1])
#%%
#Check crop and image edge detection
def cropper(seq,x1,x2,y1,y2,singleimage=False):
    if seq.ndim==2:
        return seq[y1:y2, x1:x2]
    else:
        return seq[:, y1:y2, x1:x2]

imsequence2=cropper(imsequence[-2:],15,791,701,920)
#subtract off background and invert
plt.imshow(imsequence2[0])


#%%
#Check thresholding parameters

#Import colors for plotting
import matplotlib.colors as mcolors
colors = [(0,1,0,c) for c in np.linspace(0,1,100)]
cmapg = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=5)

testimage=cropper(imsequence[800],15,791,701,920)
testback=cropper(background,15,791,701,920)

def edgedetector(inimage,background,threshval,obsSize,cannysigma):
    '''
    This function finds the edges of a cropped image of a pipette and droplet
    Image must be black on a brighter backdrop. Returns the result as a numpy
    array type object
    Arguments are:
        inimage: The input image
        background: A background image (or a 0 image)
        threshval: Threshold to use for binary thresholding of image, may be negative
        obSize: Maximum size of noise object to remove, choose smallest value that removes dust
        cannysigma: Choose value of guassian blur for edge detection
    '''
    #Subtract background if needed and select image, droplet should be high so invert
    imsub=background-inimage

    
    threshimage=imsub>threshval
    #Fill holes
    threshimage=morph2.binary_fill_holes(threshimage)
    #Remove specs
    threshimage=morph.remove_small_objects(threshimage,obsSize)
    #Find the edges
    edgedetect=feature.canny(threshimage, sigma=cannysigma)
    return edgedetect

edgedetect=edgedetector(testimage,testback,-100,20,.05)
#Plot to see
plt.imshow(testimage,cmap=plt.cm.gray)
plt.imshow(edgedetect, cmap=cmapg)
plt.tight_layout()

#%%
#Test curve fit from edges
edgearray=np.argwhere(edgedetect)

def circle(x,a,b,r):
    return np.sqrt(r**2-(x-a)**2)+b



def slopeptline(x,m,x0,y0):
    return m*(x-x0)+y0


def splitlinefinder(locs,centerybuff):

    #Which area to use for splitting line (ie how far up y)
    splitlineavregion=np.max(locs[:,0])-centerybuff
    splitline=np.mean(locs[:,1][locs[:,0]>splitlineavregion])
    return splitline

def edgeinfofinder(locs,left,pixelbuff,circfitguess,zweight):
    '''
    This function takes a numpy array of edge location xy values and returns
    the location of the contact point as well as a fitted circle
    Parameters:
        locs: the input array
        left: True for left side of droplet, false for right
        pixelbuff: How many pixels in x to include for fit
        circfitguess: Guess's for circle fit parameters, make sure to make negative for
            right side. [xcenter,ycenter,radius]
        zweight: anything below 1 gives extra weight to the zero
    
    Circle fitting can be a bit buggy, need to be fairly close with parameters.
    Better to overestimate radius somewhat.
    '''
    
    #Get the min or max position
    if left==True:
        contactloc=np.argmin(locs[:,1])
    else:
        contactloc=np.argmax(locs[:,1])
    
    contactx=locs[contactloc,1]
    contacty=locs[contactloc,0]
    
    #Set up trimmed Data set for fit using buffered area and only positive values
    #Will need to change to also include data from reflection
    if left==True:
        conds=np.logical_and(locs[:,1]<contactx+pixelbuff,locs[:,0]>contacty)
    else:
        conds=np.logical_and(locs[:,1]>contactx-pixelbuff,locs[:,0]>contacty)
        
    trimDat=locs[conds]-[contacty,contactx]
    
    #Set up weighting
    sigma = np.ones(len(trimDat[:,0]))
    sigma[np.argmin(trimDat[:,0])] = zweight
    #The fitter is annoyingly dependant on being close to the actual parameters values to get a good guess
    popt, pcov = curve_fit(circle, trimDat[:,1], trimDat[:,0],p0=circfitguess, sigma=sigma,maxfev=5000)
    def paramcirc(x):
        return circle(x,*popt)
    mcirc=derivative(paramcirc,0)
    #Return angle in degrees
    thet=np.arctan(mcirc)*180/np.pi
    #Shift circle back
    popt=popt+[contactx,contacty,0]
    return [contactx,contacty,popt,thet,mcirc]


#Plot
#Get data
leftedgedatatest=edgeinfofinder(edgearray,True,30,[50,10,100],1)
rightedgedatatest=edgeinfofinder(edgearray,False,15,[-50,10,100],1)

#Circles
x=np.linspace(leftedgedatatest[0],rightedgedatatest[0],100)

plt.plot(x,circle(x,*leftedgedatatest[2]))
plt.plot(x,circle(x,*rightedgedatatest[2]))

#Slope lines
xline1=np.linspace(leftedgedatatest[0]-10,leftedgedatatest[0]+15,100)
plt.plot(xline1,slopeptline(xline1,leftedgedatatest[-1],leftedgedatatest[0],leftedgedatatest[1]))

xline2=np.linspace(rightedgedatatest[0]-15,rightedgedatatest[0]+10,100)
plt.plot(xline2,slopeptline(xline2,rightedgedatatest[-1],rightedgedatatest[0],rightedgedatatest[1]))

plt.plot(edgearray[:,1],edgearray[:,0],'.',markersize=3)
plt.axes().set_aspect('equal')


#%%


#%%
'''
If in seperate files
import glob
imfilenames=glob.glob("Images/*.png")
numFiles=len(imfilenames)

   
timeLocArray=np.zeros([numFiles,2])
for i in range(numFiles):
    timeLocArray[i]=endfind(imfilenames[i],"background.png",10,100,460,690)
    
'''

#%%
#Crop all images
imsequence=cropper(imsequence,15,791,701,920)
background=cropper(background,15,791,701,920,True)

#%%
plt.imshow(imsequence[-1])
#%%
#Run analysis across images
numIm=imsequence[:,0,0].size
'''
leftx=np.zeros(numIm.size)
rightx=np.zeros(numIm.size)
thetr=np.zeros(numIm.size)
thetl=np.zeros(numIm.size)
center=np.zeros(numIm.size)
'''

dataarray=np.zeros([numIm,5])

for i in range(numIm):
    #Get image type array of edges
    edgedetect=edgedetector(imsequence[i],background,-100,5,.05)
    #Convert to array with location of nonzero points
    edgearray=np.argwhere(edgedetect)
    #Find the center points of the top of the pipette
    dataarray[i,0]=splitlinefinder(edgearray,15)
    #Get info on the droplet and write to lists
    leftedgedatatemp=edgeinfofinder(edgearray,True,30,[50,10,100],1)
    rightedgedatatemp=edgeinfofinder(edgearray,False,15,[-50,10,100],1)
    dataarray[i,1]=leftedgedatatemp[0]
    dataarray[i,2]=rightedgedatatemp[0]
    dataarray[i,3]=leftedgedatatemp[3]
    dataarray[i,4]=rightedgedatatemp[3]
    
    
    

np.save("testanalysisdat",dataarray)
#%%
runDat=np.load("testanalysisdat.npy")
plt.subplots(3,1,figsize=(5,8))
ax1=plt.subplot(3, 1, 1)
ax1.plot(runDat[:,1],label='left')
ax1.plot(runDat[:,2],label='right')
ax1.set_ylabel("position")
ax1.legend()

ax2=plt.subplot(3, 1, 2)
ax2.plot(runDat[:,2]-runDat[:,1])
ax2.set_ylabel("difference")
ax2.set_ylim(100,150)

ax3=plt.subplot(3, 1, 3)
ax3.plot(runDat[:,3])
ax3.plot(-runDat[:,4])
ax3.set_ylabel("angle (degrees)")
ax3.set_xlabel("time (s)")
ax3.set_ylim(60,80)


plt.tight_layout()
#%%
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

centerpos=runDat[:,1]

smoothed=running_mean(centerpos, 41)
vel=np.gradient(smoothed)
plt.plot(smoothed,vel)
plt.xlabel('position')
plt.ylabel('velocity')
#%%
plt.plot(smoothed[100:],vel[100:])
plt.ylabel("velocity")
plt.xlabel("position")
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