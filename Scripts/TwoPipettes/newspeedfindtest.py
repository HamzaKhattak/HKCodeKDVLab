# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 10:23:59 2023

@author: WORKSTATION
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.morphology as morph
from scipy.signal import find_peaks
import scipy.ndimage.morphology as morph2
from skimage import feature
from scipy.signal import savgol_filter
import pynumdiff as pynd
from skimage.io import imread as imread2
import pickle
#%%
def findcenter(raw_image,threshtype = 0,h_edge = (40,1),v_edge = (1,25)):
	
	if threshtype == 0:
		thresh_image = cv2.threshold(raw_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]	
	
	if threshtype == 1:
		thresh_image = cv2.threshold(raw_image,105,255,cv2.THRESH_BINARY_INV)[1]
		
	#Fill holes and get rid of any specs
	filling_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
	#Add pixel border to top and bottom to make holes at extrema are filled
	#Fill holes
	thresh_image=morph2.binary_fill_holes(thresh_image,filling_kernel)
	#Remove specs
	thresh_image=morph.remove_small_objects(thresh_image,500).astype('uint8')
	
	#Detect horizontal and vertical lines, only keep ones with sufficient vertical bits
	# Remove horizontal lines
	#horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
	#detected_lines = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
	
	# Add back the 
	vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,25))
	result = 1 - cv2.morphologyEx(1 - thresh_image, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)
	result=result.astype(bool)
	drop_points = np.argwhere(result)
	y0 = np.mean(drop_points[:,0])
	x0 = np.mean(drop_points[:,1])
	edgedetect=feature.canny(result, sigma=2)
	locs_edges=np.flip(np.argwhere(edgedetect),1)
	x1=dropstartendfind(raw_image[400])
	
	return x0,y0, locs_edges, x1

#202	309	1065	851	598

newim = imread2('run_2_MMStack_Pos0.ome.tif')
newim = newim[:,202:1065,309:851]
plt.imshow(newim[0])

#%%
testcenter1 = newim[0,400].astype(np.int8)
testcenter2 = newim[1,400].astype(np.int8)
plt.plot(testcenter2-testcenter1)
#%%
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
'''
takes in two arrays and finds the cross correlation array of shifts
'''

def crosscorrelator(a,b):
	'''
	This function takes in two 1D arrays a and b, normalizes them
	to find the cross correlation of a with b and then returns and
	returns an [x,y] list with the index of 
	'''
	#Normalize the input vectors
	norma = (a - np.mean(a)) / (np.std(a) * len(a))
	normb = (b - np.mean(b)) / (np.std(b))
	#Use numpy correlate to find the correlation
	corry = np.correlate(norma, normb, 'full')
	#Shift back to find the x array
	corrx = np.arange(2*len(norma)-1)-(len(norma)-1)
	return np.transpose([corrx,corry])

def gaussfunc(x,a,mu,sig):
	return a*np.exp((-(x-mu)**2)/(2*sig))


def centerfinder(vecx,vecy,buff):
	'''
	This function takes a 1D vector and fits a gaussian to its max
	peak. The buff (an integer) argument decides how many points to use around the
	max value
	'''
	#Find where the max peak is generally
	maxpos=np.argmax(vecy)
	#Find the 2 edges to use for this fit and trim the data
	lefte=maxpos-buff
	righte=maxpos+buff
	xdata=vecx[lefte:righte]
	ydata=vecy[lefte:righte]
	#Perform the curve fit, guess parameters, just since maxpos
	# will be pretty close to the center
	popt, pcov = curve_fit(gaussfunc,xdata,ydata,p0=[1,vecx[maxpos],2*buff])
	#Find standard deviation in parameters
	perr = np.sqrt(np.diag(pcov))
	#Return parameter and standard deviation
	return popt, perr

def xvtfinder(images,baseimage,cutloc,gausspts1):
	'''
	Takes a image sequence and the original image and returns series of shifts
	as well as the full cross correlation arrays
	from the base image using cross correlation at the y pixel defined by cutloc
	gaussspts1 is the number of points to use in the gaussian fit on either side
	'''
	imdim=images.ndim
	#Account for single image case
	if imdim==2:
		images=np.expand_dims(images, 0)

	#Create empty arrays to store data
	centerloc=np.zeros([images.shape[0],2])
	alldat=np.zeros([images.shape[0],images.shape[2]*2-1,2])
	#autocorrelation for base
	basecut=baseimage[cutloc]
	basecorr=crosscorrelator(basecut,basecut)
	bgparam, bgerr = centerfinder(basecorr[:,0],basecorr[:,1],gausspts1)
	#Perform cross correlation and use gaussian fit to find center position
	for i in range(images.shape[0]):
		alldat[i] = crosscorrelator(images[i,cutloc],basecut)
		gparam, gerr = centerfinder(alldat[i,:,0],alldat[i,:,1],gausspts1)
		centerloc[i]=[gparam[1],gerr[1]]
	#Account for the 0 point
	centerloc = centerloc-[bgparam[1],0]
	return centerloc, alldat


testlocs, scratch = xvtfinder(newim,newim[0],400,5)


#%%
def dropstartendfind(inputline):
	flipped = np.max(inputline)-inputline
	flippedsmooth = savgol_filter(flipped, 11, 3)
	diffs = np.abs(np.diff(flippedsmooth))
	plt.plot(flippedsmooth)
	maxdiff = np.max(diffs)
	peaklocs = find_peaks(diffs,height=.3*maxdiff)[0]
	backval = peaklocs[0]
	frontval = peaklocs[-1]
	print(frontval)
	center = (backval+frontval)/2
	return center
dropstartendfind(testcenter)
#%%
numFrames = newim.shape[0]
x0=np.zeros(numFrames)
y0=np.zeros(numFrames)
x1=np.zeros(numFrames)

for i in range(numFrames):
	x0[i], y0[i],scratch, x1[i] = findcenter(newim[i])
#%%
plt.plot(np.gradient(x0),label='old')
plt.plot(np.gradient(x1),label='new')
plt.plot(np.gradient(testlocs[:,0]),label='cc')
plt.legend()