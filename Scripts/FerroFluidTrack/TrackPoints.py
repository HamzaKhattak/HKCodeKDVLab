# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 13:54:27 2024

@author: hamza
"""

import numpy as np
import pickle, os, sys, importlib
import tifffile as tf
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series  # for convenience

import pims
import trackpy as tp

#%%



#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"F:\ferro\Experiments\Concentration05\Pip3\multidrop4_1"

#Use telegram to notify
tokenloc = r"F:\ferro\token.txt"


os.chdir(CodeDR) #Set  current working direcotry to the code directory


sys.path.append('./Scripts/FerroFluidTrack') #Add the tools to the system path so modules can be imported

#Import required modules
import PointFindFunctions as pff
importlib.reload(pff)

import FrametoTimeAndField as fieldfind
importlib.reload(fieldfind)

import NNfindFunctions as nnfind
importlib.reload(nnfind)

#Remove to avoid cluttering path
sys.path.remove('./Scripts/FerroFluidTrack') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)


#%%
params = pff.openparams('InputRunParams.txt')
run_name = params['run_name']
fieldfind.findGuassVals(params['fieldspath'], params['inputimage'],params['GaussSave'])

#Import the images of interest and a base image for background subtraction
tifobj = tf.TiffFile(params['inputimage'])
numFrames = len(tifobj.pages)
ims =  tf.imread(params['inputimage'],key=slice(0,numFrames))
background = tf.imread(params['backgroundim'])

#Import the data
locations = pff.openlistnp('initialrunpositions.pik')

gaussvals = np.loadtxt('FrametoGauss.csv',delimiter=',')[:,2]

#%%

'''
This is the section of code to track individual droplets if needed
'''
def converttoDataFrame(inputlocationarray):
	'''
	Trackpy likes the input to be a Pandas dataframe ()
	'''
	frame = np.array([],dtype=float)
	x= np.array([],dtype=float)
	y=np.array([],dtype=float)
	#convert to dataframe:
	for i in range(len(inputlocationarray)):
		frame = np.append(frame, i*np.ones(len(inputlocationarray[i])))
		x = np.append(x,inputlocationarray[i][:,0])
		y = np.append(y,inputlocationarray[i][:,1])
	
	
	pddat = pd.DataFrame({'frame': frame, 'x': x, 'y': y})
	return pddat

posdataframe = converttoDataFrame(locations)

#%%

#This section of code is for trackpy if it is needed
t1 = tp.link(posdataframe, 10,memory=50)
#%%

t2 = tp.filter_stubs(t1,50)
print('Before:', t1['particle'].nunique())
print('After:', t2['particle'].nunique())
d = tp.compute_drift(t2)

tm = tp.subtract_drift(t2.copy(), d)
tp.plot_traj(t2)
ax = tp.plot_traj(tm)
plt.show()
#%%
from scipy.signal import savgol_filter 
t4 = t2.loc[t2['particle'] == 22]
print(len(t4))
t4smooth = savgol_filter(t4.x, 11, 3)
plt.plot(t4.x,'-')

plt.plot(t4smooth,'.')
#%%
nunique = t1['particle'].nunique()
ind = [None]*nunique

t3 = t2.copy()
for i in np.arange(0,nunique,1):
	ind[i] = t3.loc[t3['particle'] == i]
	xsmooth = savgol_filter(ind[i].x, 31, 3)
	ysmooth = savgol_filter(ind[i].y, 31, 3)
	t3.loc[t3['particle'] == i,'x']=xsmooth
	t3.loc[t3['particle'] == i,'y']=ysmooth

#%%

t4 = t2.loc[t2['particle'] == 22].x
t5= t3.loc[t2['particle'] == 22].x
plt.plot(t4)
plt.plot(t5)
#%%

newlist = [t3.loc[t3['frame'] == i,['x','y']].to_numpy() for i in range(len(ims))]

pff.savelistnp('testsmoothpositions.pik', newlist)

