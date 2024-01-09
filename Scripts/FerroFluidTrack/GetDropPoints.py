# -*- coding: utf-8 -*-
"""
This code outputs droplet positions (unorganized) from a video
This code is meant to run in Spyder so you can zoom in to 
"""

import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv
from scipy.optimize import curve_fit

import imageio, os, importlib, sys, time


from datetime import datetime
from matplotlib import colors




#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\hamza\Documents\GitHub\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"C:\Users\hamza\OneDrive\Research\FerroFluids\MagnetInitialAnalysis"


os.chdir(CodeDR) #Set  current working direcotry to the code directory


sys.path.append('./Scripts/FerroFluidTrack') #Add the tools to the system path so modules can be imported

#Import required modules
from PointFindFunctions import *
#Remove to avoid cluttering path
sys.path.remove('./Scripts/FerroFluidTrack') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)

#%%


#Import the images of interest and a base image for background subtraction
ims = imageio.imread('multimages.tif')
background = cv.imread('base.tif',0)

#%%
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
		#matches[j], positions[j], ws,hs = findpositions(correctedims[i],
		#												templates[j],masks[j],
		#												ccorr_thresholds[j],
		#												ccminsep,
		#
		#										meth='cv.TM_CCOEFF_NORMED')
		
		matches[j], positions[j], ws,hs = findpositionstp(correctedims[i],
														templates[j],masks[j],
														ccorr_thresholds[j],
														ccminsep,
		
												meth='cv.TM_CCOEFF_NORMED')		
		
		shift[j] = [ws//2,hs//2]
		
		#Remove duplicates, strating from the laste template
		if j!=0:			
			for k in range(j):
				positions[j] = removeduplicates(positions[k], positions[j], compareminsep)
				#Currently just delete in order of npumber, could maybe do intensity comparison
			refinedpositions[j] = refinelocations(matches[j],positions[j]-shift[j],4)+shift[j]
		else:
			refinedpositions[j] = refinelocations(matches[j],positions[j]-shift[j],4)+shift[j]
		
		
		plt.plot(positions[j][:,1],positions[j][:,0],'.')
		plt.plot(refinedpositions[j][:,1],refinedpositions[j][:,0],'r.')
	combopositions = np.concatenate(refinedpositions,axis=0)



#%%

templatemetadata = {'crops': crops,'maskthresholds': mask_thresholds,'ccorthresh': ccorr_thresholds,'minD': [ccminsep,compareminsep]}

np.save(run_name+'metadata.npy',templatemetadata)


#%%
allpositions = [None]*len(correctedims)
allrefinedpositions = [None]*len(correctedims)
#%%

from win11toast import notify


def findoneframepositions(im,templates,masks,ccorr_thresholds,ccminsep,compareminsep):
	matches=[None]*len(mask_thresholds)
	positions=[None]*len(mask_thresholds)
	refinedpositions =[None]*len(mask_thresholds)
	shift = [None]*len(mask_thresholds)
	for j in range(len(mask_thresholds)):
		matches[j], positions[j], ws,hs =  findpositionstp(allims[i],
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
	
	
	mask_thresholds = analysisparams['maskthresholds']
	ccorr_thresholds = analysisparams['ccorthresh']
	ccminsep, compareminsep = analysisparams['minD']
	
	allpositions=[None]*len(allims)
	allpositions=[None]*len(allims)
	
	for i in  range(len(allims)): #Full run of the above but without plotting etc
		matches=[None]*len(mask_thresholds)
		positions=[None]*len(mask_thresholds)
		refinedpositions =[None]*len(mask_thresholds)
		shift = [None]*len(mask_thresholds)
		for j in range(len(mask_thresholds)):
			matches[j], positions[j], ws,hs =  findpositionstp(allims[i],
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
				
		allpositions[i], allrefinedpositions[i] = np.concatenate(positions,axis=0)	
		allrefinedpositions[i] = np.concatenate(refinedpositions,axis=0)
		
		if reportfreq ==True:
			if i%reportfreq==0:
				t2 = time.time()
				spf = (t2-t0)/reportfreq
				print('Image {imnum}, at {speed:4.4f} sec/frame'.format(imnum=i, speed=spf))
				t0=t2
		return allpositions, allrefinedpositions


#%%
test1,test2 = fullpositionfind(correctedims[:4], templates, masks, templatemetadata)
#%%
savelistnp(run_name+'positions.pik',allpositions)


#%%

#%%

import matplotlib.animation as animation
fig,ax = plt.subplots()
#line, = ax.plot([], [], lw=2)
im=ax.imshow(correctedims[0],cmap='gray')
#points, = ax.plot(allrefinedlocs[0][:,1],allrefinedlocs[0][:,0],'.')
points, = ax.plot(allrefinedpositions[0][:,1],allpositions[0][:,0],'.')
# initialization function: plot the background of each frame
# initialization function: plot the background of each frame
def init():
    im.set_data(correctedims[0])
	
    return im,points,

# animation function.  This is called sequentially
def animate_func(i):
	im.set_array(correctedims[i])
	#points.set_data(allrefinedlocs[i][:,1],allrefinedlocs[i][:,0])
	points.set_data(allrefinedpositions[i][:,1],allrefinedpositions[i][:,0])
	#points.set_data(test2.y[i],test2.x[i])
	return im,points,

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = len(correctedims),
                               interval = 1,blit=True, # in ms
                               )