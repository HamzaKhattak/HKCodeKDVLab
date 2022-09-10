# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 16:54:11 2022

@author: hamza
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio as imo
import cv2
import skimage.morphology as morph
from scipy.signal import find_peaks
import scipy.ndimage.morphology as morph2
from scipy.ndimage import label as ndimlabel2
from scipy.ndimage import sum as ndimsum2
from skimage import feature
#%%
from skimage.io import imread as imread2

allimages = imread2('DropMoves.tif')[:,600:900]
raw_image = allimages[-1]
#plt.imshow(raw_image,cmap='gray')
#%%

def findcenter(raw_image,threshtype = 0,h_edge = (40,1),v_edge = (1,25)):
	
	if threshtype == 0:
		thresh_image = cv2.threshold(raw_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]	
	
	if threshtype == 1:
		thresh_image = cv2.threshold(raw_image,105,255,cv2.THRESH_BINARY_INV)[1]
		
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
	drop_points = np.argwhere(result)
	y0 = np.mean(drop_points[:,0])
	x0 = np.mean(drop_points[:,1])
	edgedetect=feature.canny(result, sigma=2)
	locs_edges=np.flip(np.argwhere(edgedetect),1)
	return x0,y0, locs_edges


def imagesplit(image,centerx,centery,edge):
	'''
	Splits the image to give sections of the pipette
	'''
	leftend = np.int16(np.min([edge[:,0]]))-20
	rightend = np.int16(np.max([edge[:,0]]))+20
	centeryint = np.int16(centery)
	
	if centerx<0: #Just use left edge for now
		rightshift = rightend
		top = image[:centeryint,rightend:]
		bottom = image[centeryint:,rightend:]
	else:
		rightshift = 0
		top = image[:centeryint,:leftend]
		bottom = image[centeryint:,:leftend]
	return top, bottom, [leftend,rightend,rightshift,centeryint]


def pipettesplitold(pipimage):
	'''
	Outputs lines representing the edge of two pipettes from an image of pipettes
	'''
	thresh_image = cv2.threshold(pipimage, 95, 255, cv2.THRESH_BINARY_INV)[1]
	thresh_image = morph.remove_small_objects(thresh_image.astype(bool),20).astype('uint8')
	skelim = morph.skeletonize(thresh_image)
	
	topvals = np.zeros(pipimage.shape[1])
	botvals = np.zeros(pipimage.shape[1])
	xarray = np.arange(pipimage.shape[1])

	for i in range(pipimage.shape[1]):
		slicelocs = np.argwhere(skelim[:,i])
		if len(slicelocs) == 2:
			if slicelocs[1]-slicelocs[0] >2:
				topvals[i] , botvals[i] = slicelocs
		
	toparray = np.array([xarray[topvals!=0],topvals[topvals!=0]])
	botarray = np.array([xarray[botvals!=0],botvals[botvals!=0]])
	return toparray, botarray

def pipettesplit(pipimage):
	'''
	Outputs lines representing the edge of two pipettes from an image of pipettes
	'''
	topvals = np.zeros(pipimage.shape[1])
	botvals = np.zeros(pipimage.shape[1])
	
	for i in range(pipimage.shape[1]):
		flipped = np.max(pipimage[:,i])-pipimage[:,i]
		flippedsmooth =savgol_filter(flipped, 11, 3)
		diffs = np.abs(np.diff(flippedsmooth))
		diffs = diffs/np.max(diffs)
		peaklocs = find_peaks(diffs,height=.3)[0]
		topvals[i] = peaklocs[0]
		botvals[i] = peaklocs[-1]
	

	xarray = np.arange(pipimage.shape[1])
		
	toparray = np.array([xarray,topvals])
	botarray = np.array([xarray,botvals])
	return toparray, botarray

def linedefs(topline,botline,shifts):
	'''
	returns the centerline of a pipette given the location of its edges
	the shifts put the data back into the original image form
	'''
	shiftx = shifts[0]
	shifty = shifts[1]
	topfit = np.polyfit(topline[0]+shiftx, topline[1]+shifty, 1)
	botfit = np.polyfit(botline[0]+shiftx, botline[1]+shifty, 1)
	pipwdith = topfit[1]-botfit[1]
	centerline = np.mean([topfit,botfit],axis=0)
	return centerline

def paramfind(upperline,lowerline,centerx):
	'''
	finds the angle (in deg) between the pipettes given the centerlines of each pipette
	finds the seperation distance and distance to center
	'''
	m1 = upperline[0]
	m2 = lowerline[0]
	thet1 = np.abs(np.arctan(m1))
	thet2 = np.abs(np.arctan(m2))
	angle = thet1+thet2

	uline= np.poly1d(upperline)
	lline= np.poly1d(lowerline)
	sep_distance = np.abs(uline(centerx)-lline(centerx))
	
	d_to_pipcenter = sep_distance/(2*np.tan(angle/2))
	
	return angle, sep_distance, d_to_pipcenter


x0, y0, testloc_edges = findcenter(raw_image)

plt.figure(figsize=(15,3))
plt.imshow(raw_image,cmap='gray')
plt.plot(testloc_edges[:,0],testloc_edges[:,1],'.')
plt.plot(x0,y0,'ro')
#%%
itest = 0
topim, botim, splitparams = imagesplit(allimages[0], xlocs[0], ylocs[0], edges[0])
topim2, botim2, splitparams2 = imagesplit(allimages[-1], xlocs[-1], ylocs[-1], edges[-1])
plt.plot(topim[:,0])
plt.plot(topim2[:,0])
plt.plot(botim[:,0])
plt.plot(botim2[:,0])
testsplit = pipettesplit(topim)

#%%
plt.imshow(topim-topim2[:],cmap='gray')
#%%
xlocs = np.zeros(len(allimages))
ylocs = np.zeros(len(allimages))
edges = [None]*(len(allimages))

upper_line_params = np.zeros((len(allimages),2))
lower_line_params = np.zeros((len(allimages),2))

points = [None]*len(allimages)

pip_angles = np.zeros(len(allimages))
sep_distances = np.zeros(len(allimages))
d_to_centers = np.zeros(len(allimages))

for i in range(len(allimages)):
	xlocs[i] , ylocs[i], edges[i] = findcenter(allimages[i])
	topim, botim, splitparams = imagesplit(allimages[i], xlocs[i], ylocs[i], edges[i])

	upper_points = pipettesplit(topim)
	lower_points = pipettesplit(botim)
	points[i] = upper_points,lower_points
	upper_line_params[i] = linedefs(upper_points[0],upper_points[1],[splitparams[2],0])
	lower_line_params[i] = linedefs(lower_points[0], lower_points[1], [splitparams[2],splitparams[-1]])

	pip_angles[i], sep_distances[i], d_to_centers[i] = paramfind(upper_line_params[i],lower_line_params[i],xlocs[i])


#%%
from scipy.signal import savgol_filter
smoothdist = savgol_filter(d_to_centers,51,3)
plt.plot(d_to_centers,'.')
plt.plot(smoothdist)
#%%
plt.plot(xlocs,np.gradient(xlocs))
#%%
plt.plot(sep_distances,'.')
smoothsep = savgol_filter(sep_distances,51,3)
plt.plot(smoothsep)

#%%
plt.plot(pip_angles,'.')
smoothangles = savgol_filter(pip_angles,151,3)
plt.plot(smoothangles)

xsmooth = smoothsep/(2*np.tan(smoothangles/2))
speeds = np.abs(np.gradient(xsmooth))


#Single set of lines to find positions
meanangle = np.mean(pip_angles)
uline= np.poly1d(upper_line_params[0])
lline= np.poly1d(lower_line_params[0])
sep_d2 = np.abs(uline(xlocs)-lline(xlocs))

xone = sep_d2/(2*np.tan(meanangle/2))
xsmooth2 = savgol_filter(xone, 31, 3)


plt.plot(xsmooth,speeds,'b-')
plt.plot(xsmooth2,np.abs(np.gradient(xsmooth2)),'r-')

#%%
plt.plot(diffs)
#%%
diffimage = allimages[0].astype(int)-allimages[-1].astype(int)
plt.imshow(diffimage,cmap='gray')
plt.plot()
#%%

from matplotlib import animation
from matplotlib_scalebar.scalebar import ScaleBar
dt=13.6



xarray=np.arange(1600)

# ax refers to the axis propertis of the figure
fig, ax = plt.subplots(2,1,figsize=(8,6), gridspec_kw={'height_ratios': [1, 1]})
im = ax[1].imshow(allimages[0],cmap=plt.cm.gray,aspect='equal')
scalebar = ScaleBar(2.25e-6,frameon=False,location='lower right') # 1 pixel = 0.2 meter


ax[0].plot(xsmooth*2,speeds)
ax[0].set_xlabel('distance from center ($\mu m$)')
ax[0].set_ylabel('speed')

ax[1].axis('off')
ax[1].get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
ax[1].get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis

edgeline, = ax[1].plot(edges[0][:,0],edges[0][:,1],color='red',marker='.',linestyle='',markersize=1,animated=True)

topline, = ax[1].plot(xarray,np.poly1d(upper_line_params[0])(xarray),color='cyan',marker='.',linestyle='',markersize=1,animated=True)
botline, = ax[1].plot(xarray,np.poly1d(lower_line_params[0])(xarray),color='cyan',marker='.',linestyle='',markersize=1,animated=True)

currentspeed, = ax[0].plot([], [],'ro', animated=True)
centerpoint, =  ax[1].plot([], [],'ro', animated=True)





def init():
	"""
	This function gets passed to FuncAnimation.
	It initializes the plot axes
	"""
	#Set plot limits etc
	ax[1].add_artist(scalebar)


	#plt.tight_layout()
	return centerpoint,edgeline,currentspeed,topline,botline,
#fig.tight_layout(pad=0)

def update_plot(it):
	#global xAnim, yAnim
	
	#This this section plots the force over time
	edgeline.set_data(edges[it][:,0],edges[it][:,1])
	topline.set_data(xarray,np.poly1d(upper_line_params[it])(xarray))
	botline.set_data(xarray,np.poly1d(lower_line_params[it])(xarray))
	centerpoint.set_data(xlocs[it],ylocs[it])
	currentspeed.set_data(xsmooth[it]*2,speeds[it])
	#Plot of image
	im.set_data(allimages[it])
	
	return centerpoint,im,edgeline,currentspeed,topline,botline,
plt.tight_layout()

#Can control which parts are animated with the frames, interval is the speed of the animation
# now run the loop
ani = animation.FuncAnimation(fig, update_plot, frames=np.arange(0,len(allimages)), interval=50,
                    init_func=init, repeat_delay=1000, blit=True)


#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

plt.show()

#%%
Writer = animation.writers['ffmpeg']
writer = Writer(fps=10,extra_args=['-vcodec', 'libx264'])
ani.save('speedtracking.mp4',writer=writer,dpi=200)
