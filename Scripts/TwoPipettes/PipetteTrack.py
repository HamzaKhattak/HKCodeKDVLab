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
testim=fullseqimport('mainmergevid.ome.tif')



template = imageio.imread('crosscorr.tif')
w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

#%%
import scipy.ndimage.morphology as morph
from scipy import ndimage
inter = template<120
inter2 = morph.binary_fill_holes(inter)

struct2 = ndimage.generate_binary_structure(2, 2)

inter2 = ndimage.binary_dilation(inter2,structure=struct2,iterations = 4)
inter2 = np.array(inter2,dtype=np.uint8)
plt.imshow(inter2)
#%%
 # Apply template Matching
locs = np.zeros((len(testim),2))

for i in range(len(testim)):
	res = cv.matchTemplate(testim[i],template,cv.TM_CCOEFF_NORMED,mask = inter2)
	min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
	top_left = max_loc
	locs[i] = top_left
	if i%200==0:
		print(i)
#%%
np.save('ratchetpositions.npy',locs)
#%%
plt.plot(locs[:,0]-locs[0,0])
plt.plot(locs[:,1]-locs[0,1])
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

#%%
import pyometiff

from pyometiff import OMETIFFReader


reader = OMETIFFReader(fpath='mainmergevid.ome.tif')

img_array, metadata, xml_metadata = reader.read()

import re
keywords = ["DeltaT=", "humidity", "pressure"]
keyword = 'DeltaT='
match = re.findall(f"{keyword}.*?(\d+[.]\d+)", xml_metadata)

match = np.array(match,dtype=float)
np.save('timesmerge.npy',match)

#%%
import cv2
import scipy.ndimage as morph2
from skimage import feature
import skimage.morphology as morph
def findthresh(raw_image,threshtype = 0,h_edge = (40,1),v_edge = (1,25),threshold = 50):
	
	if threshtype == 0:
		thresh_image = cv2.threshold(raw_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]	
	
	if threshtype == 1:
		thresh_image = cv2.threshold(raw_image,threshold,255,cv2.THRESH_BINARY_INV)[1]
		
	#Fill holes and get rid of any specs
	filling_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
	#Add pixel border to top and bottom to make holes at extrema are filled
	#Fill holes
	thresh_image=morph2.binary_fill_holes(thresh_image,filling_kernel)
	#Remove specs
	thresh_image=morph.remove_small_objects(thresh_image,500).astype('uint8')
	
	#Detect horizontal and vertical lines, only keep ones with sufficient vertical bits
	# Remove horizontal lines
	#horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
	#detected_lines = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
	
	# Add back the 
	vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,25))
	result = 1 - cv2.morphologyEx(1 - thresh_image, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)
	result=result.astype(bool)
	#edgedetect=feature.canny(result, sigma=2)
	#locs_edges=np.flip(np.argwhere(edgedetect),1)
	return result

#%%
raw_image = testim[-210]
thresh_image = cv2.threshold(raw_image, 40,255, cv2.THRESH_BINARY_INV)[1]	
plt.imshow(findthresh(testim[300],1,threshold=65))
#%%



ims = np.array([findthresh(i,1,threshold=60) for i in testim])

#%%
plt.imshow(ims[0])
#%%
fig, ax = plt.subplots()
im = ax.imshow(findthresh(raw_image,1),cmap=plt.cm.gray,aspect='equal')
xdata, ydata = [], []
ln, = ax.plot([], [], 'ro')
def init():
	return ln,

def update_plot(it):
	im.set_data(ims[it])
	return im,

ani = FuncAnimation(fig, update_plot, frames=np.arange(0,len(testim)), interval=1,
                    init_func=init, repeat_delay=1000, blit=True)
plt.show()

#%%
ims = ims.astype('uint8')
contours, hierarchy = cv.findContours(ims[0], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)



# compute the center of the contour
M = cv2.moments(contours[0])
cX = M["m10"] / M["m00"]
cY = M["m01"] / M["m00"]

print(cX,cY)
x = np.mean(contours[0][:,0,0])
y = np.mean(contours[0][:,0,1])
print(x,y)
def dropsxyfind(im):
	contours, hierarchy = cv.findContours(im, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	locs = [[0,0],[0,0]]
	
	if len(contours) == 2:
		for i, cnt in enumerate(contours):
			M = cv2.moments(cnt)
			cX = M["m10"] / M["m00"]
			cY = M["m01"] / M["m00"]
			locs[i] = [cX,cY]
		if locs[0][0]>locs[1][0]:
			locs = locs[::-1]
	if len(contours) == 1:
		M = cv2.moments(contours[0])
		cX = M["m10"] / M["m00"]
		cY = M["m01"] / M["m00"]
		locs = [[cX,cY],[cX,cY]]
	if len(contours) > 2:
		contours = sorted(contours, key=cv2.contourArea, reverse=True)
		contours = contours[:1]
		for i, cnt in enumerate(contours):
			M = cv2.moments(cnt)
			cX = M["m10"] / M["m00"]
			cY = M["m01"] / M["m00"]
			locs[i] = [cX,cY]
		if locs[0][0]>locs[1][0]:
			locs = locs[::-1]
	return locs
			
			
#%%

#%%
positions = np.zeros((len(ims),2,2))
for i in range(len(ims)):
	positions[i] = dropsxyfind(ims[i])

#%%
pf = positions.reshape(len(positions),-1)
pfi = np.argwhere(pf == 0)[:,0]
pf[pfi] = (pf[pfi+1]+pf[pfi-1])/2

plt.plot(pf[:,0])
plt.plot(pf[:,2])

np.save('dropletpositions.npy',pf)


#%%
fig, ax = plt.subplots()
im = ax.imshow(findthresh(raw_image,1),cmap=plt.cm.gray,aspect='equal')
xdata, ydata = [], []
ln, = ax.plot([pf[0,0],pf[0,2]], [pf[0,1],pf[0,3]], 'ro')
def init():
	return ln,

def update_plot(it):
	im.set_data(ims[it])
	ln.set_data([pf[it,0],pf[it,2]], [pf[it,1],pf[it,3]],)
	return im,ln,

ani = FuncAnimation(fig, update_plot, frames=np.arange(0,len(testim)), interval=1,
                    init_func=init, repeat_delay=1000, blit=True)
plt.show()
