# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 20:45:10 2024

@author: hamza
"""


import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv
from scipy.optimize import curve_fit

from skimage.feature.peak import peak_local_max

import imageio
import time
from datetime import datetime
from matplotlib import colors
import pickle


import pandas as pd
from pandas import DataFrame, Series  # for convenience

import pims
import trackpy as tp

def savelistnp(filepath,data):
	'''
	Saves lists of numpy arrays using pickle so they don't become objects
	'''
	with open(filepath, 'wb') as outfile:
		   pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)

def openlistnp(filepath):
	'''
	Opens lists of numpy arrays using pickle
	'''
	with open(filepath, 'rb') as infile:
	    result = pickle.load(infile)
	return result

def rescale(data):
	'''

	rescales data to uint8

	'''
	newdat = 255*(data-np.min(data))/(np.max(data)-np.min(data))
	return newdat.astype(np.uint8)

def imagepreprocess(ims,background):
	'''
	This function takes a series of images and a background image,and applies a correction
	for non uniform lighting and returns a corrected images. The corrected image
	is scaled such that the maximum value is the maximum value in a uint8 image
	'''
	#blur the background to a large extend
	blur = cv.blur(background,(400,400))
	#subtract off, converting to int to avoid losing negatives
	correctedim = ims.astype(int)-blur.astype(int)
	#rescale back to unint8 as needed for cross correlation
	correctedim = rescale(correctedim)
	return correctedim

def templatecropper(inarray,crops):
	'''
	Simply crops a template given crop points in [x1,y1],[x2,y2] form
	'''
	return inarray[crops[0][1]:crops[1][1],crops[0][0]:crops[1][0]]



def ccor(im,template,mask,meth='cv.TM_CCOEFF_NORMED'):
	'''
	This code runs cross correlation on an image with a given template and mask
	The returned cross correlation is taken to the power of 3 and then 
	normalized to emphasize the peaks
	'''
	w, h = template.shape[::-1]
	data=np.zeros((h,w,3),dtype=np.uint8)
	initialmatch = cv.matchTemplate(im,template,eval(meth),data,mask)
	match = initialmatch**3/np.max(initialmatch**3) #^3 to emphasize the peaks, could change if needed
	return match, w, h

def findpositions(im,template,mask,threshold,minD, meth='cv.TM_CCOEFF_NORMED',removethresh = 150):
	'''
	This code uses the scipy peak_local_max to find the locations of peaks in 
	the droplet images. It first runs the cross-correlation to get the input
	for find peaks.  Need a minimum theshold cutoff to define what is a peak 
	and a minimum distance so as not to overcount peaks
	The function returns  the cross correlation image, peaks and w and height of
	the inputted tempate

	'''
	match, w, h = ccor(im,template,mask,meth)
	peaks = peak_local_max(match, min_distance=minD,threshold_abs=threshold) #find peaks
	peakbrightness = im[peaks[:,0]+w//2,peaks[:,1]+h//2] #find brightness at peak locations
	peaks = peaks[peakbrightness<removethresh] #Only keep peaks where image is dark
	peaks = peaks + [w//2,h//2] #shift to correct location
	return match, peaks, w, h

def findpositionstp(im,template,mask,threshold,minD, peaksize = 3, peaksizecut = 0.003, percentilecut = 0.01, meth='cv.TM_CCOEFF_NORMED', removethresh = 150):
	'''
	This code uses trackpy locate  to find the locations of peaks in 
	the droplet images. It first runs the cross-correlation to get the input
	for find peaks.  Need a minimum theshold cutoff to define what is a peak 
	and a minimum distance so as not to overcount peaks
	The function returns  the cross correlation image, peaks and w and height of
	the inputted tempate

	'''
	match, w, h = ccor(im,template,mask,meth)
	match = np.clip(match,0,np.inf) #Better to just clip out any negatives
	positions = tp.locate(match,peaksize,minmass=peaksizecut,separation=minD,percentile=percentilecut,invert=True)
	positions = positions.loc[positions['size']>0]
	positions = np.transpose([positions.y+w//2,positions.x+h//2])
	intlocs = positions.astype(int)
	peakbrightness = im[intlocs[:,0],intlocs[:,1]] #find brightness at peak locations
	positions = positions[peakbrightness<removethresh] #Only keep peaks where image is dark
	return match, positions, w, h

def distances(xy1, xy2):
	'''
	Finds distance between points in two arrays Nx2 arrays of points
	
	'''
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
	x = np.linspace(-windowsize,windowsize,2*windowsize+1,dtype=int)
	y = x
	X = np.meshgrid(x, y)

	#First crop to small section around the peak
	for i in range(len(initiallocs)):
		yc = int(initiallocs[i,0])
		xc = int(initiallocs[i,1])
		imsize = inputccor.shape
		condition =  (xc < imsize[1]-(windowsize+1) and xc > (windowsize+1) and
				yc<imsize[0]-(windowsize+1) and yc > (windowsize+1))
		if (condition):
			cropped = inputccor[yc-windowsize:yc+windowsize+1 , xc-windowsize:xc+windowsize+1]
			#initial_guess = (cropped[windowsize,windowsize],windowsize,windowsize-1,windowsize-1,windowsize,0,cropped[0,0])
			initial_guess = (10,0,0,cropped[0,0])
			inputdata = np.ravel(cropped)
			bnds=((-np.inf, -.5, -.5, -np.inf), (np.inf, .5, .5, np.inf))
			popt, pcov = curve_fit(twoD_power2,X,inputdata,p0=initial_guess,maxfev=1000,bounds=bnds)
			
			'''
			if i ==100:
				plt.figure()
				plt.imshow(cropped,cmap='gray')
				plt.plot(windowsize,windowsize,'bo')
				plt.plot(popt[1]+windowsize,popt[2]+windowsize,'ro')
				print(popt)
			'''
			locs[i] = popt[1]+yc,popt[2]+xc
		else:
			locs[i] = [yc,xc]
	return locs


def findoneframepositions(im,templates,masks,ccorr_thresholds,ccminsep,compareminsep):
	matches=[None]*len(ccorr_thresholds)
	positions=[None]*len(ccorr_thresholds)
	refinedpositions =[None]*len(ccorr_thresholds)
	shift = [None]*len(ccorr_thresholds)
	for j in range(len(ccorr_thresholds)):
		matches[j], positions[j], ws,hs =  findpositionstp(im,
														templates[j],masks[j],
														ccorr_thresholds[j],
														ccminsep,
		
												meth='cv.TM_CCOEFF_NORMED')	
		shift[j] = [ws//2,hs//2]
		if j!=0:			
			for k in range(j):
				positions[j] = removeduplicates(positions[k], positions[j], compareminsep)
				#Currently just delete in order of number, could maybe do intensity comparison
			refinedpositions[j] = refinelocations(matches[j],positions[j]-shift[j],4)+shift[j]
		else:
			refinedpositions[j] = refinelocations(matches[j],positions[j]-shift[j],4)+shift[j]
	positions = np.concatenate(positions,axis=0)	
	refinedpositions = np.concatenate(refinedpositions,axis=0)
	return positions,refinedpositions			
				
def fullpositionfind(allims,templates,masks,analysisparams,report =True, reportfreq =100):
	t0=time.time()
	
	ccorr_thresholds = analysisparams['ccorthresh']
	ccminsep, compareminsep = analysisparams['minD']
	
	allpositions=[None]*len(allims)
	allrefinedpositions=[None]*len(allims)
	
	for i in  range(len(allims)): #Full run of the above but without plotting etc
				
		allpositions[i], allrefinedpositions[i] = findoneframepositions(allims[i],templates,masks,ccorr_thresholds,ccminsep,compareminsep)
		
		if reportfreq ==True:
			if i%reportfreq==0:
				t2 = time.time()
				spf = (t2-t0)/reportfreq
				print('Image {imnum}, at {speed:4.4f} sec/frame'.format(imnum=i, speed=spf))
				t0=t2
	return allpositions, allrefinedpositions