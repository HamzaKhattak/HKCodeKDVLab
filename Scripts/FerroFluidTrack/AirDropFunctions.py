# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 21:18:36 2024

@author: WORKSTATION
"""

import skimage.morphology as morph
import numpy as np
import cv2 as cv
from scipy import signal

from scipy.optimize import curve_fit
import scipy.ndimage as morph2

def cropper(inarray,crops):
	'''
	Simply crops a template given crop points in [x1,y1],[x2,y2] form
	'''
	if inarray.ndim == 3:
		r = inarray[:,crops[0][1]:crops[1][1],crops[0][0]:crops[1][0]]
	if inarray.ndim ==2:
		r = inarray[crops[0][1]:crops[1][1],crops[0][0]:crops[1][0]]
	return r

def findthresh(raw_image,threshtype = 0,h_edge = (40,1),v_edge = (1,40),threshold = 50):
	
	if threshtype == 0:
		thresh_image = cv.threshold(raw_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]	
	
	if threshtype == 1:
		thresh_image = cv.threshold(raw_image,threshold,255,cv.THRESH_BINARY_INV)[1]
	#Fill holes and get rid of any specs
	filling_kernel = cv.getStructuringElement(cv.MORPH_RECT, (10,10))
	#Add pixel border to top and bottom to make holes at extrema are filled
	#Fill holes
	thresh_image=morph2.binary_fill_holes(thresh_image,filling_kernel)
	#Remove specs
	thresh_image=morph.remove_small_objects(thresh_image,500).astype('uint8')
	
	# Add back the 
	vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, v_edge)
	result = 1 - cv.morphologyEx(1 - thresh_image, cv.MORPH_CLOSE, vertical_kernel, iterations=1)
	return result


def twodropsxyfind(im):
	contours, hierarchy = cv.findContours(im, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	locs = [[0,0],[0,0]]
	if len(contours) == 2:
		for i, cnt in enumerate(contours):
			M = cv.moments(cnt)
			cX = M["m10"] / M["m00"]
			cY = M["m01"] / M["m00"]
			locs[i] = [cX,cY]
		if locs[0][0]>locs[1][0]:
			locs = locs[::-1]
	if len(contours) == 1:
		M = cv.moments(contours[0])
		cX = M["m10"] / M["m00"]
		cY = M["m01"] / M["m00"]
		locs = [[cX,cY],[cX,cY]]
	if len(contours) > 2:
		contours = sorted(contours, key=cv.contourArea, reverse=True)
		contours = contours[:2]
		for i in [0,1]:
			M = cv.moments(contours[i])
			cX = M["m10"] / M["m00"]
			cY = M["m01"] / M["m00"]
			locs[i] = [cX,cY]
		if locs[0][0]>locs[1][0]:
			locs = locs[::-1]
	return locs,contours





def gauss(x, H, A, x0, sigma): 
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def normify(x):
	#Converts the dark pipette image into a normalized image where the pipette is bright
	r = -x
	r = r-np.min(r)
	r=r/np.max(r)
	return r


def findpipcenters(cropped):
	#Finds the pipette center locations for a given image crop
	#Uses blurring
	h, w = cropped.shape
	locs = np.zeros(h)
	blurred = cv.blur(cropped,(20,20))
	x = np.arange(w)
	for i in range(h):
		y = normify(blurred[i])
		maxloc = np.argmax(y)
		po, pcov = curve_fit(gauss, x, y,p0=[0,1,maxloc,20])
		locs[i] = po[2]
	return locs

def findshifted(ims,croppoints):
	#finds the locations of the points of the pipette in the uncropped image
	cropped = cropper(ims,croppoints)
	y = np.arange(croppoints[0][1],croppoints[1][1])
	xs = np.zeros((ims.shape[0],cropped.shape[1]))
	for i in range(len(ims)):
		locs = findpipcenters(cropped[i])
		xs[i] = locs+croppoints[0][0]
	return y, xs



def findcents(x,y,cutpoint):
	'''
	Extends a line from the fitted points to find the center location of the droplet
	Given a location to cutoff for the droplet
	'''
	fitparams = np.polyfit(y,x,1)
	x0 = np.polyval(np.poly1d(fitparams), cutpoint)
	return cutpoint,x0








def ccor(a,b,meth='cv.TM_CCOEFF_NORMED'):
	'''
	This code runs cross correlation on an image with a given template
	'''
	#Normalize the input vectors
	norma = (a - np.mean(a)) / (np.std(a))
	normb = (b - np.mean(b)) / (np.std(b))
	w, h = a.shape[::-1]
	match = signal.correlate(norma, normb, mode='full', method='auto')
	match=match/np.max(match)
	#Getting shifts
	#corrx = np.arange(2*norma.shape[1]-1)-(norma.shape[1]-1)
	#corry = np.arange(2*norma.shape[0]-1)-(norma.shape[0]-1)
	shiftx = norma.shape[1]-1
	shifty = norma.shape[0]-1
	return shiftx,shifty,match



def findmaxloc(im):
	#Returns the index of the maximum point in an array
	return np.unravel_index(im.argmax(), im.shape)



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

def refinelocations(inputccor,windowsize):
	#Create the meshgrid
	initiallocs = findmaxloc(inputccor)
	x = np.linspace(-windowsize,windowsize,2*windowsize+1,dtype=int)
	y = x
	X = np.meshgrid(x, y)

	yc = initiallocs[0]
	xc = initiallocs[1]
	
	cropped = inputccor[yc-windowsize:yc+windowsize+1 , xc-windowsize:xc+windowsize+1]
	#initial_guess = (cropped[windowsize,windowsize],windowsize,windowsize-1,windowsize-1,windowsize,0,cropped[0,0])
	initial_guess = (1,0,0,10,50,0,0.05)
	inputdata = np.ravel(cropped)
	
	popt, pcov = curve_fit(twoD_Gaussian,X,inputdata,p0=initial_guess,maxfev=2000)

	locs = popt[2]+yc,popt[1]+xc
	return locs


def getshifts(croppedims):
	shiftarray=np.zeros((len(croppedims),2))
	for i in range(len(croppedims)):
		sx,sy,corrim1 = ccor(croppedims[i],croppedims[0])
		refineloc = refinelocations(corrim1,10)
		shiftarray[i] = [refineloc[0]-sy,refineloc[1]-sx]
	return shiftarray