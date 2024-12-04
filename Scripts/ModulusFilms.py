# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:06:12 2019

@author: WORKSTATION
"""

import sys, os
import matplotlib.pyplot as plt
import numpy as np
import importlib
from scipy.optimize import curve_fit
import pandas as pd
import imageio
import tifffile as tf
#import similaritymeasures

#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"F:\ShaganaFilms\MainRun10k1\Run1stretch_1"

filepath = r"F:\ShaganaFilms\MainRun10k1\Run1stretch_1\Run1stretch_MMStack_Pos0.ome.tif"

os.chdir(CodeDR) #Set  current working direcotry to the code directory


sys.path.append('./Tools') #Add the tools to the system path so modules can be imported

#Import required modules
import Crosscorrelation as crco
importlib.reload(crco)
import ImportTools as ito 
importlib.reload(ito)

#Remove to avoid cluttering path
sys.path.remove('./Tools') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)

#%%

def preimport(FilePath):
	'''
	This function creates a tifffile object that can be referenced for image import operations
	Simply a renaming of the tifffile package to keep it seperate
	This object is more of a reference to the file and has info like number of pages etc
	'''
	return tf.TiffFile(FilePath)

def singlesliceimport(FilePath,ival):
	'''
	This imports only a single frame of a tiff sequence
	'''
	tifobj = preimport(FilePath)
	return tifobj.pages[ival].asarray()


def fullseqimport(FilePath):
	'''
	This object imports the entire sequence of tiff images
	'''
	tifobj = preimport(FilePath)
	numFrames = len(tifobj.pages)
	return tf.imread(FilePath,key=slice(0,numFrames))

preim = tf.TiffFile(r'F:\ShaganaFilms\InitialSample0k\0-1umsSrun1\run_MMStack_Pos0.ome.tif')
numFrames = len(preim.pages)
#%%
from pyometiff import OMETIFFReader
reader = OMETIFFReader(fpath=filepath)

img_array, metadata, xml_metadata = reader.read()
#%%

#%%
points =img_array[:,0,0]
plt.plot(points)
#%%
endpoint=879
#%%
plt.imshow(img_array[0])

#%%
dats = img_array[:endpoint,41]

plt.plot(dats[0])
plt.plot(dats[-1])
#%%
#Get the image path names


base=dats[0]
#%%

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


xytest = crosscorrelator(dats[600], dats[0])
loctest= centerfinder(xytest[:,0],xytest[:,1],6)[0][1]
plt.plot(xytest[:,0],xytest[:,1],'.')
plt.axvline(loctest)
#%%
plt.plot(dats[0])
plt.plot(dats[-1])
#%%
locs = np.zeros(len(dats[:,0]))
for i,dat in enumerate(dats):
	xytest = crosscorrelator(dat, dats[0])
	loctest= centerfinder(xytest[:,0],xytest[:,1],6)[0][1]
	locs[i] =loctest
	
#%%
k0=2.58 #N/m for the calibration pipette
mppix = .4504e-6 
width = 3.22e-3
length = 15.53e-3
thickness = 9e-6

xs=np.linspace(0, len(locs),num=len(locs))
ys = locs*mppix

machinerange = 600e-6
ysmachine = np.linspace(0, machinerange,num=len(locs))

plt.plot(xs,ys,label='measured')
plt.plot(xs,ysmachine,label='machine')

#%%
stress=(ys-ys[0])*k0/(width*thickness)
strain = (ysmachine-ys)/length

cstress = stress[100:-100]
cstrain = strain[100:-100]
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
plt.plot(cstrain,cstress,'.')
plt.xlabel('$e$')
plt.ylabel('$\sigma$ (Pa)')
def linfx(x,a,b):
	return a*x+b

popt,perr = curve_fit(linfx,cstrain,cstress)
print(popt/1e6)
