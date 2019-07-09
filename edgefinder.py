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
    if singleimage:
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
    threshimage=morph.remove_small_objects(threshimage,5)
    #Find the edges
    edgedetect=feature.canny(threshimage, sigma=.05)
    return edgedetect

edgedetect=edgedetector(imsequence[-1],background,-100,5,.05)
#Plot to see
plt.imshow(imsequence[-1],cmap=plt.cm.gray)
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
        
    '''
    
    if left==True:
        contactloc=np.argmin(locs[:,1])
    else:
        contactloc=np.argmax(locs[:,1])
    
    contactx=locs[contactloc,1]
    contacty=locs[contactloc,0]
    
    #Only bit different
    if left==True:
        conds=np.logical_and(locs[:,1]<contactx+pixelbuff,locs[:,0]>contacty)
    else:
        conds=np.logical_and(locs[:,1]>contactx-pixelbuff,locs[:,0]>contacty)
        
    trimDat=locs[conds]-[contacty,contactx]
    
    #Set up weighting
    sigma = np.ones(len(trimDat[:,0]))
    sigma[np.argmin(trimDat[:,0])] = zweight
    #The fitter is annoyingly dependand on being close to the actual parameters values to get a good guess
    popt, pcov = curve_fit(circle, trimDat[:,1], trimDat[:,0],p0=circfitguess, sigma=sigma,maxfev=5000)
    def paramcirc(x):
        return circle(x,*popt)
    mcirc=derivative(paramcirc,0)
    #Return angle in degrees
    thet=np.arctan(mcirc)*180/np.pi
    #Shift circle back
    popt=popt+[contactx,contacty,0]
    return [contactx,contacty,popt,thet,contactx]


#Plot
leftedgedatatest=edgeinfofinder(edgearray,True,30,[50,10,100],1)
rightedgedatatest=edgeinfofinder(edgearray,False,15,[-50,10,100],1)

x=np.linspace(leftedgedatatest[0],rightedgedatatest[0],100)
plt.plot(x,circle(x,*leftedgedatatest[2]))
plt.plot(x,circle(x,*rightedgedatatest[2]))
plt.plot(edgearray[:,1],edgearray[:,0],'.',markersize=3)
plt.axes().set_aspect('equal')

#%%
trimDatMaxr=trimDatright[:,1].max()
maxPlotCirc=trimDatMaxr+50
x=np.linspace(0,maxPlotCirc,100)
plt.plot(locs[:,1],locs[:,0],'.',markersize=3)
plt.plot(x+leftx,paramcirc(x)+lefty)

p0=[50,10,100]
plt.axvline(leftx)
plt.axvline(rightx)
plt.axes().set_aspect('equal')
#%%
30
20

#%%


#%%

plt.plot(leftdat[:,1],leftdat[:,0],'.',markersize=2)
plt.plot(leftdat[minLoc,1],leftdat[minLoc,0],'o',markersize=3)
#%%
'''
old
oRem=morph.remove_small_objects(threshimage,100)
hRem=morph.remove_small_holes(oRem,100)
plt.imshow(hRem,cmap=plt.cm.gray)

edgedetect=feature.canny(hRem, sigma=.05)
plt.imshow(edgedetect, cmap=plt.cm.gray)
locs=np.argwhere(edgedetect)
'''
#%%
#Crop
imsequence=cropper(imsequence,15,791,701,920)
background=cropper(background,15,791,701,920,True)

#%%

plt.plot(locs[:,1],locs[:,0],'.',markersize=1)
plt.xlim(0,300)
plt.axes().set_aspect('equal')
plt.tight_layout()

#%%

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

def endfind2(im,imthresh,obsize,xminval,xmaxval):
    '''
    This function will return the start and end of a droplet
    main is the string indicating the location of the main file and back
    is the string for the background
    imthresh is the threshold value applied
    #ymin and ymax are used to select the portion of the image to include
    '''
    #Subtract the background and apply the threshold
    imsub=-im
    threshimage=imsub>imthresh
    
    #Remove dust specs
    oRem=morph.remove_small_objects(threshimage,obsize)
    hRem=morph.remove_small_holes(oRem,obsize)
    
    #Find the edges
    edgedetect=feature.canny(hRem, sigma=.05)
    #Convert to xy values
    locs=np.argwhere(edgedetect)
    
    #Select the correct region
    interval=np.logical_and(locs[:,1]>xminval,locs[:,1]<xmaxval)
    impDat=locs[interval]

    
    leadedge=np.max(impDat[:,0])
    trailedge=np.min(impDat[:,0])
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


#%%
plt.imshow(im[0])
#%%
numIm=im[:,0,0].size
timeLocArray=np.zeros([numIm,2])
for i in range(numIm):
    timeLocArray[i]=endfind2(im[i],-5,100,700,933)
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