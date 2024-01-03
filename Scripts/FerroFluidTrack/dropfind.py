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
	

img = cv.imread('testim3.tif',0)
background = cv.imread('base.tif',0)

imdim_y, imdim_x = img.shape

correctedim = imagepreprocess(img,background)

plt.imshow(correctedim,cmap = 'gray')
#%%
xyct = [[749,742],[763,757]] #cropping for template 1 (main)
xyct2 = [[696,633],[706,642]] #cropping for template 2 (secondary)

def templatecropper(inarray,crops):
	return inarray[crops[0][1]:crops[1][1],crops[0][0]:crops[1][0]]



template1 = templatecropper(correctedim,xyct)
template2 = templatecropper(correctedim,xyct2)
mask1=template1<150
mask1=mask1.astype(np.float32)

mask2=template2<70
mask2=mask2.astype(np.float32)
#plt.imshow(template1)
#%%
# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']


def findpositions(im,template,mask,threshold,meth='cv.TM_CCOEFF_NORMED'):
	w, h = template.shape[::-1]
	data=np.zeros((h,w,3),dtype=np.uint8)
	initialmatch = cv.matchTemplate(im,template,eval(meth),data,mask)
	match = initialmatch**3/np.max(initialmatch**3) #^3 to emphasize the peaks, could change if needed
	peaks = peak_local_max(match, min_distance=7,threshold_abs=.05) #find peaks
	peakbrightness = im[peaks[:,0]+w//2,peaks[:,1]+h//2] #find brightness at peak locations
	peaks = peaks[peakbrightness<150] #Only keep peaks where image is dark
	peaks = peaks + [w//2,h//2] #shift to correct location
	return peaks
	
positions1 = findpositions(correctedim,template1,mask1,.05,meth='cv.TM_CCOEFF_NORMED')
positions2 = findpositions(correctedim,template2,mask2,.1,meth='cv.TM_CCOEFF_NORMED')



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

allpositions = np.concatenate([positions1,positions2],axis=0)
positions2 = removeduplicates(positions1, positions2, 8)

plt.imshow(correctedim,cmap='gray')
plt.plot(positions1[:,1],positions1[:,0],'.')
plt.plot(positions2[:,1],positions2[:,0],'.')

#%%
