# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 15:33:38 2023

@author: WORKSTATION
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pims


import skimage.morphology as morph
from scipy import ndimage
from matplotlib import animation


import scipy.ndimage as morph2
from skimage import feature

allimages = np.array(pims.PyAVReaderIndexed('maintwist.avi'))[:,:,:,0]
pixsize = 2.24e-6
insecperframe = .2

#%%
def findthresh(raw_image,threshtype = 0,h_edge = (40,1),v_edge = (1,25),threshold = 50):
	
	if threshtype == 0:
		thresh_image = cv2.threshold(raw_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]	
	
	if threshtype == 1:
		thresh_image = cv2.threshold(raw_image,threshold,255,cv2.THRESH_BINARY_INV)[1]
	#Fill holes and get rid of any specs
	filling_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
	#Add pixel border to top and bottom to make holes at extrema are filled
	#Fill holes
	thresh_image=morph2.binary_fill_holes(thresh_image,filling_kernel)
	#Remove specs
	thresh_image=morph.remove_small_objects(thresh_image,500).astype('uint8')
	
	# Add back the 
	vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, v_edge)
	result = 1 - cv2.morphologyEx(1 - thresh_image, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)
	return result


plt.imshow(findthresh(allimages[100],1,threshold=60))
#%%
ims = np.array([findthresh(i,1,threshold=60) for i in allimages])
#%%
plt.imshow(ims[200])

#%%



def twodropsxyfind(im):
	contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
		contours = contours[:2]
		for i in [0,1]:
			M = cv2.moments(contours[i])
			cX = M["m10"] / M["m00"]
			cY = M["m01"] / M["m00"]
			locs[i] = [cX,cY]
		if locs[0][0]>locs[1][0]:
			locs = locs[::-1]
	return locs


			
#%%

testi = 74

contours, hierarchy = cv2.findContours(ims[testi], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
contours = sorted(contours, key=cv2.contourArea, reverse=True)
plt.plot(contours[0][:,0,0],contours[0][:,0,1])
M = cv2.moments(contours[0])
cX = M["m10"] / M["m00"]
cY = M["m01"] / M["m00"]
plt.plot(cX,cY,'ro')
#%%
positions = np.zeros((len(ims),2,2))

for i in range(len(ims)):
	positions[i] = twodropsxyfind(ims[i])
	

pf = positions.reshape(len(positions),-1)
pfi = np.argwhere(pf == 0)[:,0]
pf[pfi] = (pf[pfi+1]+pf[pfi-1])/2


np.save('twistpositions.npy',pf)

plt.plot(pf[:,0])
#%%
fig, ax = plt.subplots()
im = ax.imshow(ims[0],cmap=plt.cm.gray,aspect='equal')
xdata, ydata = [], []
ln, = ax.plot([], [], 'ro')
def init():
	return ln,

def update_plot(it):
	im.set_data(ims[it])
	ln.set_data(pf[it,[0,2]],pf[it,[1,3]])
	return im,ln,

ani = animation.FuncAnimation(fig, update_plot, frames=np.arange(850,len(ims)), interval=10,
                    init_func=init, repeat_delay=1000, blit=True)
plt.show()

#%%

#%%
fig, ax = plt.subplots()
im = ax.imshow(findthresh(ims[0],1),cmap=plt.cm.gray,aspect='equal')
xdata, ydata = [], []
ln, = ax.plot([pf[0,0],pf[0,2]], [pf[0,1],pf[0,3]], 'ro')
def init():
	return ln,

def update_plot(it):
	im.set_data(ims[it])
	ln.set_data([pf[it,0],pf[it,2]], [pf[it,1],pf[it,3]],)
	return im,ln,

ani = animation.FuncAnimation(fig, update_plot, frames=np.arange(0,len(ims)), interval=1,
                    init_func=init, repeat_delay=1000, blit=True)
plt.show()
#%%
np.save('onedroppositions.npy',pf)