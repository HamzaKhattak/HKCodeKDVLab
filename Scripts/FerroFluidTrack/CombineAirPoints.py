# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 10:18:45 2024

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

import PointFindFunctions as pff
importlib.reload(pff)

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

alldat=[None]*len(runnums)
#%%
'''
Get all the crop locations and frame to Guass data for all of the images
'''
for i, fold in enumerate(folderpaths):
	os.chdir(fold)
	alldat[i] = np.load('run3locs.npy')
os.chdir(dataDR)

#%%

pff.savelistnp('maindat.npy', alldat)