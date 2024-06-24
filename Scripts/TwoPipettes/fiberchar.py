# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:32:14 2021

@author: Hamza
"""

'''
This code performs the edge location and cross correlation analysis across multiple images
'''

import sys, os, time
import matplotlib.pyplot as plt
import numpy as np
import importlib
import imageio
#import similaritymeasures

#%%
#Specify the location of the Tools folder
CodeDR=r"F:\TrentDrive\Research\KDVLabCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"F:\TrentDrive\Research\Fibers\codetest"


os.chdir(CodeDR) #Set  current working direcotry to the code directory


sys.path.append('./Tools') #Add the tools to the system path so modules can be imported

#Import required modules
import EdgeDetection as ede
importlib.reload(ede)
import PlateauAnalysis as planl
importlib.reload(planl)

#Remove to avoid cluttering path
sys.path.remove('./Tools') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)


#%%

testimage = imageio.imread('main.tif')

edges = ede.edgedetector(testimage,False,-20,500,2)
plt.imshow(testimage,cmap='gray')
#plt.plot(edges[:,0],edges[:,1],'r.')

#%%
oneslice=testimage[:,19]

from scipy.signal import savgol_filter, butter, filtfilt
from scipy.signal import filtfilt



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y
smootheddat = np.convolve(oneslice, np.ones(4)/4, mode='valid')
smootheddat2 = butter_lowpass_filtfilt(smootheddat, 20, 300)

def find_roots(x,y):
    s = np.abs(np.diff(np.sign(y))).astype(bool) #find sign change
    return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1) #linear interp

def find_roots2(y):
	s = np.abs(np.diff(np.sign(y))).astype(bool) #find sign change
	x =  np.where(s) + 1/(np.abs(y[1:][s]/y[:-1][s])+1) #linear interp
	return x.flatten()
def getallroots(im,val):
	roots = np.zeros((im.shape[1],2))
	for i in range(im.shape[1]):
		yvals = find_roots2(im[:,i]-val)
		roots[i]=[yvals[0],yvals[-1]]

#%%
oneslice=testimage[:,19]
oneslice2=oneslice.astype(np.int16)
testimage2=testimage.astype(np.int16)
xvals=find_roots2(oneslice2-20)
plt.plot(oneslice)
plt.plot(xvals,np.ones(len(xvals))*20,'go')
#%%
im=testimage.astype(np.int)
toproots = np.zeros(im.shape[1])
botroots = np.zeros(im.shape[1])
for i in range(im.shape[1]):
	yvals = find_roots2(im[:,i]-25)
	#yvals=yvals.flatten()
	toproots[i]=yvals[-1]
	botroots[i]=yvals[0]
	
plt.imshow(testimage,cmap='gray')
plt.plot(toproots,'r')
plt.plot(botroots,'r')
#%%
plt.plot(np.diff(smootheddat2),'.')



