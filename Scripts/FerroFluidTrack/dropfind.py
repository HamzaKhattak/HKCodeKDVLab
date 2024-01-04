# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 21:26:51 2024

@author: hamza
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:48:53 2021

@author: Hamza
"""

import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv
from scipy.optimize import curve_fit

from skimage.feature.peak import peak_local_max

def rescale(data):
	'''

	rescale data to uint8

	'''
	newdat = 255*(data-np.min(data))/(np.max(data)-np.min(data))
	return newdat.astype(np.uint8)

def imagepreprocess(im,background):
	'''
	This function takes a file location, inputs an image and applies a correction
	for non uniform lighting and returns a corrected image. The corrected image
	is scaled such that the maximum value is the maximum value in a uint8 image
	'''
	#blur the background to a large extend
	blur = cv.blur(background,(400,400))
	#subtract off, converting to int to avoid losing negatives
	correctedim = im.astype(int)-blur.astype(int)
	#rescale back to unint8 as needed for cross correlation
	correctedim = rescale(correctedim)
	return correctedim

import imageio
ims = imageio.imread('multimages.tif')
background = cv.imread('base.tif',0)
blur = cv.blur(background,(400,400))

#subtract off, converting to int to avoid losing negatives
correctedims = ims.astype(int)-blur.astype(int)
fullmax = np.max(correctedims)
fullmin = np.min(correctedims)
newdat = 255*(correctedims-fullmin)/(fullmax-fullmin)
mainims = newdat.astype(np.uint8)

img = cv.imread('testim3.tif',0)
background = cv.imread('base.tif',0)

imdim_y, imdim_x = img.shape

correctedim = imagepreprocess(img,background)

plt.imshow(correctedim,cmap = 'gray')
#%%
xyct = [[749,742],[763,757]] #cropping for template 1 (main)
xyct2 = [[696,633],[709,645]] #cropping for template 2 (secondary)

def templatecropper(inarray,crops):
	return inarray[crops[0][1]:crops[1][1],crops[0][0]:crops[1][0]]



template1 = templatecropper(mainims[0],xyct)
template2 = templatecropper(mainims[0],xyct2)
mask1=template1<150
mask1=mask1.astype(np.float32)

mask2=template2<70
mask2=mask2.astype(np.float32)
plt.imshow(mask2)
#%%
# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

def ccor(im,template,mask,meth='cv.TM_CCOEFF_NORMED'):
	w, h = template.shape[::-1]
	data=np.zeros((h,w,3),dtype=np.uint8)
	initialmatch = cv.matchTemplate(im,template,eval(meth),data,mask)
	match = initialmatch**3/np.max(initialmatch**3) #^3 to emphasize the peaks, could change if needed
	return match, w, h

def findpositions(im,template,mask,threshold,meth='cv.TM_CCOEFF_NORMED'):
	match, w, h = ccor(im,template,mask,meth)
	peaks = peak_local_max(match, min_distance=7,threshold_abs=.05) #find peaks
	peakbrightness = im[peaks[:,0]+w//2,peaks[:,1]+h//2] #find brightness at peak locations
	peaks = peaks[peakbrightness<150] #Only keep peaks where image is dark
	peaks = peaks + [w//2,h//2] #shift to correct location
	return match, peaks, w, h

match1, positions1, w1,h1 = findpositions(correctedim,template1,mask1,.05,meth='cv.TM_CCOEFF_NORMED')
match2, positions2, w2,h2 = findpositions(correctedim,template2,mask2,.1,meth='cv.TM_CCOEFF_NORMED')



def distances(xy1, xy2):
   d0 = np.subtract.outer(xy1[:,0], xy2[:,0])
   d1 = np.subtract.outer(xy1[:,1], xy2[:,1])
   return np.hypot(d0, d1)

def removeduplicates(main,secondary,minsepdistance):
	'''
	Removes any duplicates from second array
	'''
	dvals = distances(main,secondary)
	remove = np.argwhere(dvals<minsepdistance)[:,1]
	return np.delete(secondary,remove,axis=0)

combopositions = np.concatenate([positions1,positions2],axis=0)
positions2 = removeduplicates(positions1, positions2, 8)

plt.imshow(correctedim,cmap='gray')
plt.plot(positions1[:,1],positions1[:,0],'.')
plt.plot(positions2[:,1],positions2[:,0],'.')

#%%
allpositions=[None]*len(mainims)

for i in range(100):
	positions1 = findpositions(mainims[i],template1,mask1,.03,meth='cv.TM_CCOEFF_NORMED')
	positions2 = findpositions(mainims[i],template2,mask2,.03,meth='cv.TM_CCOEFF_NORMED')
	positions2 = removeduplicates(positions1, positions2, 8)
	combopositions = np.concatenate([positions1,positions2],axis=0)
	allpositions[i] = combopositions
#%%

#%%
import pandas as pd
from pandas import DataFrame, Series  # for convenience




import pims
import trackpy as tp
tp.link()
#%%


def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
	
    return g.ravel()

def twoD_power2(xy, a, xo, yo, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    g = offset - ((x-xo)**2 +(y-yo)**2)/a**2
	
    return g.ravel()
def refinelocations(inputccor,initiallocs,windowsize):
	#Create the meshgrid
	locs = np.zeros([len(initiallocs),2])
	x = np.linspace(0,2*windowsize-1,2*windowsize,dtype=int)
	y = x
	X = np.meshgrid(x, y)

	#First crop to small section around the peak
	for i in range(len(initiallocs)):
		yc = initiallocs[i,0]
		xc = initiallocs[i,1]
		cropped = inputccor[yc-windowsize:yc+windowsize , xc-windowsize:xc+windowsize]
		#initial_guess = (cropped[windowsize,windowsize],windowsize,windowsize-1,windowsize-1,windowsize,0,cropped[0,0])
		initial_guess = (1,windowsize,windowsize,cropped[0,0])
		inputdata = np.ravel(cropped)
		bnds=((-np.inf, windowsize-2, windowsize-2, 0), (np.inf, windowsize+2, windowsize+2, 200))
		popt, pcov = curve_fit(twoD_power2,X,inputdata,p0=initial_guess,maxfev=5000)
		locs[i] = popt[1]+yc-windowsize,popt[2]+xc-windowsize
	return locs

testim,initiallocs,w1,h1 = findpositions(mainims[0],template1,mask1,.05,meth='cv.TM_CCOEFF_NORMED')
initiallocs = initiallocs-[w1//2,h1//2]
test = refinelocations(testim,initiallocs,3)
test = test+[w1//2,h1//2]
initiallocs = initiallocs +[w1//2,h1//2]
'''
plt.imshow(mainims[0],cmap='gray')
plt.plot(test[:,1],test[:,0],'r.')
plt.plot(initiallocs[:,1],initiallocs[:,0],'b.')
'''
#%%
yc = initiallocs[80,0]
xc = initiallocs[80,1]
plt.imshow(testim[yc-3:yc+3 , xc-3:xc+3])
#%%
testnums = 10
allrefinedlocs = [None]*testnums
for i in range(testnums):
	print(i)
	testim, w1, h1 = ccor(mainims[i],template1,mask1,meth='cv.TM_CCOEFF_NORMED')
	testim,initiallocs,w1,h1 = findpositions(mainims[i],template1,mask1,.05,meth='cv.TM_CCOEFF_NORMED')
	initiallocs = initiallocs-[w1//2,h1//2]
	test = refinelocations(testim,initiallocs,3)
	test = test+[w1//2,h1//2]
	initiallocs = initiallocs +[w1//2,h1//2]
	allrefinedlocs[i] = test


#%%

import matplotlib.animation as animation
fig = plt.figure()
#line, = ax.plot([], [], lw=2)
im=plt.imshow(mainims[0],cmap='gray')
points, = plt.plot(allrefinedlocs[0][:,1],allrefinedlocs[0][:,0],'.')
# initialization function: plot the background of each frame
# initialization function: plot the background of each frame
def init():
    im.set_data(mainims[0])
	
    return im,points,

# animation function.  This is called sequentially
def animate_func(i):
	im.set_array(mainims[i])
	points.set_data(allrefinedlocs[i][:,1],allrefinedlocs[i][:,0])
	return im,points,

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = 10,
                               interval = 1,blit=True, # in ms
                               )
