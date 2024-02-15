# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 13:00:14 2024

@author: WORKSTATION
"""

import imageio, sys, os, importlib
import matplotlib.pyplot as plt
import tifffile as tf
import cv2 as cv
import numpy as np
from scipy import signal
import glob
from scipy.optimize import curve_fit

#%%

#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"F:/ferro/air/largedrops"


os.chdir(CodeDR) #Set  current working direcotry to the code directory


sys.path.append('./Scripts/FerroFluidTrack') #Add the tools to the system path so modules can be imported

#Import required modules
import AirDropFunctions as adf
importlib.reload(adf)

import FrametoTimeAndField as ftf
importlib.reload(ftf)


import FrametoTimeAndField as fieldfind
importlib.reload(fieldfind)

#Remove to avoid cluttering path
sys.path.remove('./Scripts/FerroFluidTrack') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)

#%%
def folderlistgen(mainfolderloc):
	'''
	This function returns a sorted list of the folders in the current working directory
	The first argument
	mainfolderloc is where the subfolders of each run from the experiment from the run are
	'''
	folderpaths=glob.glob(mainfolderloc+'/*/')
	foldernames=next(os.walk(mainfolderloc))[1]
	#filenames=glob.glob("*.tif") #If using single files
	#Sort the folders by the leading numbers
	runnums=[int(i.split('_')[1]) for i in foldernames]
	foldernames = [x for _,x in sorted(zip(runnums,foldernames))]
	folderpaths = [x for _,x in sorted(zip(runnums,folderpaths))]
	return folderpaths, foldernames, sorted(runnums)


folderpaths, foldernames, runnums = folderlistgen(os.getcwd())
#%%
os.chdir(dataDR)
guassnames = ['run'+str(i)+'.csv' for i in runnums]
gausssaves = ['guassconvert'+str(i) for i in runnums]
croplocs = [None]*len(runnums)
fielddat=[None]*len(runnums)
#%%
'''
Get all the crop locations and frame to Guass data for all of the images
'''
for i, fold in enumerate(folderpaths):
	os.chdir(fold)
	fielddat[i] = ftf.findGuassVals(guassnames[i],'run_MMStack_Pos0.ome.tif',gausssaves[i])
	
	#Import the images of interest and a base image for background subtraction
	tifobj = tf.TiffFile('run_MMStack_Pos0.ome.tif')
	numFrames = len(tifobj.pages)
	im =  tf.imread('run_MMStack_Pos0.ome.tif',key=0)
	im2 = tf.imread('run_MMStack_Pos0.ome.tif',key=-200)
	plt.imshow(im,cmap='gray')
	plt.imshow(im2,cmap='gray',alpha=0.3)
	croplocs[i] = plt.ginput(4) 
	plt.close()
os.chdir(dataDR)


#%%
croplocs2 = croplocs.copy()

leftcrops = [None]*len(runnums)
rightcrops = [None]*len(runnums)
for i,crops in enumerate(croplocs2):
	croplocs2[i] = [list( map(int,i) ) for i in crops] 
	leftcrops[i] = croplocs2[i][:2]
	rightcrops[i] = croplocs2[i][2:]


#%%
for i, fold in enumerate(folderpaths):
	'''
	Import the iamges
	'''
	os.chdir(fold)
	tifobj = tf.TiffFile('run_MMStack_Pos0.ome.tif')
	numFrames = len(tifobj.pages)
	ims =  tf.imread('run_MMStack_Pos0.ome.tif',key=slice(0,numFrames))
	 
	'''
	Find center locations of droplets initially, use threshold to get initial y values
	'''
	threshims = adf.findthresh(ims[0],1,threshold=30)
	cents,contours = adf.twodropsxyfind(threshims)
	
	lefty0 = cents[0][1]
	righty0 = cents[1][1]

	'''
	Find the initial x position using pipette location
	'''
	
	#Crop the left and right pipettes
	leftcrop = leftcrops[i]
	rightcrop = rightcrops[i]
	left = adf.cropper(ims,leftcrop)
	right = adf.cropper(ims,rightcrop)
	
	
	ileftline = adf.findshifted(ims[:2], leftcrop)
	irightline = adf.findshifted(ims[:2], rightcrop)
	
	initialcenterleft = adf.findcents(ileftline[1][0],ileftline[0],lefty0)
	initialcenterright = adf.findcents(irightline[1][0],irightline[0],righty0)
	
	'''
	Find amount of shift from initial positions
	'''
	leftshifts = adf.getshifts(left)
	rightshifts =  adf.getshifts(right)
	
	leftx = leftshifts[:,1] + initialcenterleft[1]
	rightx = rightshifts[:,1] + initialcenterright[1]
	
	'''
	Save with the field and time data
	'''
	t=fielddat[i][:,1]
	gauss = fielddat[i][:,2]
	
	tosave = np.transpose([t,gauss,leftx,rightx])

	np.save('run'+str(i)+'locs.npy',tosave)
	print('run ' + str(i)+ 'done')
