# -*- coding: utf-8 -*-
"""
This code outputs droplet positions (unorganized) from a video
This code is meant to run in Spyder so you can zoom in to 
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
#%%
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
#%%

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

#Import the images of interest and a base image for background subtraction
ims = imageio.imread('multimages.tif')
background = cv.imread('base.tif',0)


correctedims = imagepreprocess(ims, background)


plt.imshow(correctedims[0],cmap='gray') #Imshow to allow cropping to find template locations


#%%

run_name = 'initialrun'

cropframes = [0,0,500]
xyct1 = [[749,742],[763,757]] #cropping for template 1 (main)
xyct2 = [[644,627],[656,640]] #cropping for template 2 (secondary)
xyct3 = [[539,491],[549,502]] #cropping for template 2 (secondary)

crops = [xyct1,xyct2,xyct3]

#Thresholds for the masks used in cross correlation input
mask_thresholds = [150,90,80]
ccorr_thresholds = [.1,.1,0.1]
ccminsep = 7    
compareminsep = 8



def templatecropper(inarray,crops):
	return inarray[crops[0][1]:crops[1][1],crops[0][0]:crops[1][0]]

c_white = colors.colorConverter.to_rgba('red',alpha = 0)
c_red= colors.colorConverter.to_rgba('red',alpha = .1)
cmap_rb = colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white,c_red],512)
#plt.imshow(thresholdfinder(correctedimage),cmap=cmap_rb)

templates = [None]*len(mask_thresholds)
masks = [None] * len(mask_thresholds)
for i in range(len(mask_thresholds)):
	templates[i] = templatecropper(correctedims[cropframes[i]],crops[i])
	masks[i] = templates[i] < mask_thresholds[i]
	masks[i]=masks[i].astype(np.float32)
	plt.figure()	
	plt.imshow(templates[i],cmap='gray')
	plt.imshow(masks[i],cmap=cmap_rb)
#%%
# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

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

def findpositions(im,template,mask,threshold,minD, meth='cv.TM_CCOEFF_NORMED'):
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
	peaks = peaks[peakbrightness<150] #Only keep peaks where image is dark
	peaks = peaks + [w//2,h//2] #shift to correct location
	return match, peaks, w, h

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
		yc = initiallocs[i,0]
		xc = initiallocs[i,1]
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
	return locs

for i in [0,500]: #Pick a couple test images
	plt.figure()
	plt.imshow(correctedims[i],cmap='gray')
	
	#Initialize empty arrays for the data
	matches=[None]*len(mask_thresholds)
	positions=[None]*len(mask_thresholds)
	refinedpositions =[None]*len(mask_thresholds)
	shift = [None]*len(mask_thresholds)
	
	#Loop over the templates
	for j in range(len(mask_thresholds)):
		matches[j], positions[j], ws,hs = findpositions(correctedims[i],
														templates[j],masks[j],
														ccorr_thresholds[j],
														ccminsep,
														meth='cv.TM_CCOEFF_NORMED')
		shift[j] = [ws//2,hs//2]
		
		#Remove duplicates, strating from the laste template
		if j!=0:			
			for k in range(j):
				positions[j] = removeduplicates(positions[k], positions[j], compareminsep)
				#Currently just delete in order of number, could maybe do intensity comparison
			refinedpositions[j] = refinelocations(matches[j],positions[j]-shift[j],4)+shift[j]
		else:
			refinedpositions[j] = refinelocations(matches[j],positions[j]-shift[j],4)+shift[j]
		
		
		plt.plot(positions[j][:,1],positions[j][:,0],'b.')
		plt.plot(refinedpositions[j][:,1],refinedpositions[j][:,0],'r.')
	combopositions = np.concatenate(refinedpositions[j],axis=0)



#%%

templatemetadata = {'crops': crops,'maskthresholds': mask_thresholds,'ccorthresh': ccorr_thresholds,'minD': [ccminsep,compareminsep]}

np.save(run_name+'metadata.npy',templatemetadata)


#%%
allpositions=[None]*len(correctedims)
t0=time.time()
for i in  range(len(correctedims)): #Full run of the above but without plotting etc
	matches=[None]*len(mask_thresholds)
	positions=[None]*len(mask_thresholds)
	refinedpositions =[None]*len(mask_thresholds)
	shift = [None]*len(mask_thresholds)
	for j in range(len(mask_thresholds)):
		matches[j], positions[j], ws,hs = findpositions(correctedims[i],
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
		
	allpositions[i] = np.concatenate(refinedpositions,axis=0)
	if i%100==0:
		t2 = time.time()
		spf = (t2-t0)/50
		print('Image {imnum}, at {speed:4.4f} sec/frame'.format(imnum=i, speed=spf))
		t0=t2

#%%

savelistnp(run_name+'positions.pik',allpositions)

#%%

import matplotlib.animation as animation
fig,ax = plt.subplots()
#line, = ax.plot([], [], lw=2)
im=ax.imshow(correctedims[0],cmap='gray')
#points, = ax.plot(allrefinedlocs[0][:,1],allrefinedlocs[0][:,0],'.')
points, = ax.plot(allpositions[0][:,1],allpositions[0][:,0],'.')
# initialization function: plot the background of each frame
# initialization function: plot the background of each frame
def init():
    im.set_data(correctedims[0])
	
    return im,points,

# animation function.  This is called sequentially
def animate_func(i):
	im.set_array(correctedims[i])
	#points.set_data(allrefinedlocs[i][:,1],allrefinedlocs[i][:,0])
	points.set_data(allpositions[i][:,1],allpositions[i][:,0])
	#points.set_data(test2.y[i],test2.x[i])
	return im,points,

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = len(correctedims),
                               interval = 1,blit=True, # in ms
                               )