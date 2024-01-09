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

from win11toast import notify



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

#Run the image correction to flatten the brighness
correctedims = imagepreprocess(ims, background)

plt.imshow(correctedims[0],cmap='gray') #Imshow to allow cropping to find template crop locations


#%%

'''
This section  of code is for getting the masks used in cross correlation
'''
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


templatemetadata = {'crops': crops,'maskthresholds': mask_thresholds,'ccorthresh': ccorr_thresholds,'minD': [ccminsep,compareminsep]}


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
'''
Run the analysis on some test images to make sure it works
'''

inims = correctedims[[0,500]]

testpos,testrpos = fullpositionfind(inims, templates, masks, templatemetadata)

plt.figure()
plt.plot(testrpos[0][:,1],testrpos[0][:,0],'.')
plt.imshow(correctedims[0],cmap='gray')
plt.figure()
plt.plot(testrpos[1][:,1],testrpos[1][:,0],'.')
plt.imshow(correctedims[500],cmap='gray')

#%%

#%%
'''
Run the analysis and save the relevant metadata
'''
np.save(run_name+'locfindmetadata.npy',templatemetadata)
allpositions, allrefinedpositions = fullpositionfind(correctedims, templates, masks, templatemetadata)


savelistnp(run_name+'positions.pik',allpositions)

notifytext = run_name + ' is done.'
notify(notifytext)

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