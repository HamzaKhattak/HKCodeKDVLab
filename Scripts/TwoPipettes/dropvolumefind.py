# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

import imageio as io

from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
testim=io.imread(filename)


fig = plt.figure('Pick top left and bottom right corner and then fit lines')
plt.imshow(testim,cmap='gray')
plt.grid(which='both')

print('Select crop points for droplet')
crop_points = np.floor(plt.ginput(2,timeout=200)) #format is [[xmin,ymin],[xmax,ymax]]
crop_points=crop_points.astype(int)

print('Select crop points for pipette')
pcrop_points = np.floor(plt.ginput(2,timeout=200)) #format is [[xmin,ymin],[xmax,ymax]]
pcrop_points=pcrop_points.astype(int)

dropim= testim[crop_points[0,1]:crop_points[1,1],crop_points[0,0]:crop_points[1,0]]
pipim = testim[pcrop_points[0,1]:pcrop_points[1,1],pcrop_points[0,0]:pcrop_points[1,0]]
plt.close()
#%%


from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.ndimage import rotate
def linefind(pipimage,x0,y0):
	'''
	Outputs lines representing the edge of two pipettes from an image of pipettes
	'''
	x1=x0
	x2=x0+pipimage.shape[1]
	xs = np.arange(x1,x2)
	
	topvals = np.zeros(pipimage.shape[1])
	botvals = np.zeros(pipimage.shape[1])
	for i in range(pipimage.shape[1]):
		flipped = np.max(pipimage[:,i])-pipimage[:,i]
		flippedsmooth = savgol_filter(flipped, 15, 3)
		diffs = np.abs(np.diff(flippedsmooth))
		maxdiff = np.max(diffs)
		peaklocs = find_peaks(diffs,height=.7*maxdiff,prominence=.1)[0]
		topvals[i] = peaklocs[0]+y0
		botvals[i] = peaklocs[-1]+y0
	
	topfit = np.polyfit(xs,topvals, 1)
	botfit = np.polyfit(xs,botvals, 1)
	pipwidth = topfit[1]-botfit[1]
	centerline = np.mean([topfit,botfit],axis=0)
	return centerline, pipwidth


rotateparams, pipettewidth = linefind(pipim,pcrop_points[0,0],pcrop_points[0,1])
testrotate = rotate(dropim,-np.arctan(rotateparams[0]))

def pipettedef(dropimage,x0,y0):
	'''
	Outputs lines representing the edge of two pipettes from an image of pipettes
	'''
	x1=x0
	x2=x0+dropimage.shape[1]
	xs = np.arange(x1,x2)
	
	topvals = np.zeros(dropimage.shape[1])
	botvals = np.zeros(dropimage.shape[1])
	for i in range(dropimage.shape[1]):
		flipped = np.max(dropimage[:,i])-dropimage[:,i]
		flippedsmooth = savgol_filter(flipped, 11, 3)
		diffs = np.abs(np.diff(flippedsmooth))
		maxdiff = np.max(diffs)
		peaklocs = find_peaks(diffs,height=.3*maxdiff,prominence=2.5,rel_height=.9)[0]
		topvals[i] = peaklocs[0]+y0
		botvals[i] = peaklocs[-1]+y0
	
	return xs, topvals, botvals


samplex = np.arange(pcrop_points[0,0], pcrop_points[0,0]+pipim.shape[1])


plt.figure(figsize=(5,4))
pipettelocs = pipettedef(dropim,crop_points[0,0],crop_points[0,1])
plt.imshow(testim,cmap='gray')
plt.plot(pipettelocs[0],pipettelocs[1],'.')
plt.plot(pipettelocs[0],pipettelocs[2],'.')

plt.plot(samplex,np.poly1d(rotateparams)(samplex),'r-') # the center line

plt.figure(figsize=(5,4))
samplex = np.arange(crop_points[0,0],crop_points[0,0]+dropim.shape[1])

centered= [pipettelocs[2]-np.poly1d(rotateparams)(samplex),pipettelocs[1]-np.poly1d(rotateparams)(samplex)]
averagepipettes = np.mean(np.abs(centered),axis=0)

plt.plot(pipettelocs[2]-np.poly1d(rotateparams)(samplex))
plt.plot(pipettelocs[1]-np.poly1d(rotateparams)(samplex))
plt.plot(averagepipettes)


pixsize = 1.78e-6
totalvolume = np.sum(np.pi*averagepipettes**2)
pipettevolume = len(averagepipettes)*np.pi*pipettewidth**2
dropvolume=totalvolume-pipettevolume
dropvolume = dropvolume*pixsize**3
print(str(dropvolume*10**12) +' picoliters')