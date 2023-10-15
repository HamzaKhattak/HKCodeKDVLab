# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 15:33:38 2023

@author: WORKSTATION
"""
import os, glob
import imageio.v2 as imageio
#need to install imageio and imagio-ffmpeg and tkinter 
import tifffile as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
def preimport(FilePath):
	'''
	This function creates a tifffile object that can be referenced for image import operations
	Simply a renaming of the tifffile package to keep it seperate
	This object is more of a reference to the file and has info like number of pages etc
	'''
	return tf.TiffFile(FilePath)
def fullseqimport(FilePath):
	'''
	This object imports the entire sequence of tiff images
	'''
	tifobj = preimport(FilePath)
	numFrames = len(tifobj.pages)
	return tf.imread(FilePath,key=slice(0,numFrames))
testim=fullseqimport('mainzigzag.ome.tif')



template = imageio.imread('crosscorr.tif')
w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

#%%
import scipy.ndimage.morphology as morph
inter = template<110
inter2 = morph.binary_fill_holes(inter)
inter2 = np.array(inter2,dtype=np.uint8)
plt.imshow(inter2)
#%%
 # Apply template Matching
locs = np.zeros((len(testim),2))

for i in range(len(testim)):
	res = cv.matchTemplate(testim[i],template,cv.TM_CCOEFF_NORMED)
	min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
	top_left = max_loc
	locs[i] = top_left
	if i%100==0:
		print(i)
	
#%%
plt.imshow(testim[200],cmap='gray')
plt.plot(locs[200][0],locs[200][1],'o')
#%%
plt.imshow(testim[324])
#%%
from scipy import signal
b, a = signal.butter(3, 0.04)
filtx = signal.filtfilt(b, a, locs[:,0])
filty = signal.filtfilt(b, a, locs[:,1])
#plt.plot(np.gradient(newdat))

plt.plot(filtx-filtx[0])
plt.plot(filty-filty[0])

#%%
b, a = signal.butter(2, 0.03)
filtx = signal.filtfilt(b, a, locs[:,0])
plt.plot(locs[:,0])
plt.plot(filtx)

#%%
plt.plot(np.gradient(filtx))
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = ax.plot([], [], 'ro')
im = ax.imshow(testim[0],cmap=plt.cm.gray,aspect='equal')

def init():
	return ln,

def update_plot(it):
	ln.set_data(locs[it,0], locs[it,1])
	im.set_data(testim[it])
	return ln,im,

ani = FuncAnimation(fig, update_plot, frames=np.arange(0,len(testim)), interval=1,
                    init_func=init, repeat_delay=1000, blit=True)
plt.show()

