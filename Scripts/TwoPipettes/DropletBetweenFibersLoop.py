# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 16:54:11 2022

@author: hamza
"""

import numpy as np
import cv2
import skimage.morphology as morph
from scipy.signal import find_peaks
import scipy.ndimage as morph2
from skimage import feature
from scipy.signal import savgol_filter
from skimage.io import imread as imread2
import pickle
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


def imagesplit(image,centerx,centery,edge,leftside=True):
	'''
	Splits the image to give sections of the pipette
	'''
	leftend = np.int16(np.min([edge[:,0]]))-20
	rightend = np.int16(np.max([edge[:,0]]))+20
	centeryint = np.int16(centery)
	
	if leftend==True: #Just use left edge for now
		rightshift = rightend
		top = image[:centeryint,rightend:]
		bottom = image[centeryint:,rightend:]
	else:
		rightshift = 0
		top = image[:centeryint,:leftend]
		bottom = image[centeryint:,:leftend]
	return top, bottom, [leftend,rightend,rightshift,centeryint]



def pipettesplit(pipimage,avoidrange):
	'''
	Outputs lines representing the edge of two pipettes from an image of pipettes
	'''
	topvals = np.zeros(pipimage.shape[1])
	botvals = np.zeros(pipimage.shape[1])
	xarray = np.arange(pipimage.shape[1])
	avoidarray = np.full(pipimage.shape[1],True)
	avoidarray[avoidrange[0]:avoidrange[1]] = False
	
	for i in range(pipimage.shape[1]):
		if avoidarray[i]==True:
			flipped = np.max(pipimage[:,i])-pipimage[:,i]
			flippedsmooth = savgol_filter(flipped, 11, 3)
			diffs = np.abs(np.diff(flippedsmooth))
			maxdiff = np.max(diffs)
			peaklocs = find_peaks(diffs,height=.3*maxdiff)[0]
			topvals[i] = peaklocs[0]
			botvals[i] = peaklocs[-1]
	
		
	toparray = np.array([xarray[avoidarray],topvals[avoidarray]])
	botarray = np.array([xarray[avoidarray],botvals[avoidarray]])
	return toparray, botarray

def linedefs(topline,botline,shifty):
	'''
	returns the centerline of a pipette given the location of its edges
	the shifts put the data back into the original image form
	'''

	topfit = np.polyfit(topline[0], topline[1]+shifty, 1)
	botfit = np.polyfit(botline[0], botline[1]+shifty, 1)
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
	angle = np.arctan(np.abs((m2-m1)/(1+m1*m2)))

	uline= np.poly1d(upperline)
	lline= np.poly1d(lowerline)
	sep_distance = np.abs(uline(centerx)-lline(centerx))
	
	d_to_pipcenter = sep_distance/(2*np.tan(angle/2))
	
	return angle, sep_distance, d_to_pipcenter



#%%




def time_series_paramfind(input_images,midpoint):

	dimensions = input_images[0].shape
	
	xlocs = np.zeros(len(input_images))
	ylocs = np.zeros(len(input_images))
	edges = [None]*(len(input_images))
	
	upper_line_params = np.zeros((len(input_images),2))
	lower_line_params = np.zeros((len(input_images),2))
	
	points = [None]*len(input_images)
	
	pip_angles = np.zeros(len(input_images))
	sep_distances = np.zeros(len(input_images))
	d_to_centers = np.zeros(len(input_images))
	
	for i in range(len(allimages)):
		xlocs[i] , ylocs[i], edges[i] = findcenter(allimages[i])
	
		leftend = np.int16(np.min([edges[i][:,0]]))-20
		if leftend <=0:
			leftend = 0
		rightend = np.int16(np.max([edges[i][:,0]]))+20
		if rightend >=dimensions[1]:
			rightend = dimensions[1]
		
		upper_points = pipettesplit(allimages[i][:midpoint],[leftend,rightend])
		lower_points = pipettesplit(allimages[i][midpoint:],[leftend,rightend])
		points[i] = upper_points,lower_points
		upper_line_params[i] = linedefs(upper_points[0],upper_points[1],0)
		lower_line_params[i] = linedefs(lower_points[0], lower_points[1], midpoint)
	
		pip_angles[i], sep_distances[i], d_to_centers[i] = paramfind(upper_line_params[i],lower_line_params[i],xlocs[i])

	return xlocs, ylocs, pip_angles, sep_distances, d_to_centers, [upper_line_params,lower_line_params], edges



#%%
runparams = np.loadtxt('runsparams.csv',skiprows=1,dtype=str,delimiter=',')
run_names = runparams[:,0]
run_time_steps = runparams[:,1].astype(float)
run_crops = runparams[:,2:].astype(float)
run_crops = run_crops.astype(int)

#%%
print('start')
for i in range(len(run_names)):
#for i in [10,11,12]:
	#Get the crop points from the run parameters
	x0=run_crops[i][0]
	x1=run_crops[i][2]
	y0=run_crops[i][1]
	y1=run_crops[i][3]
	halfpoint = run_crops[i][4]
	allimages = imread2(run_names[i])[:,y0:y1,x0:x1]
	image_params = time_series_paramfind(allimages, halfpoint-y0)
	leadtxt = run_names[i].split('.')[0]
	np.save(leadtxt+'.npy', image_params[:4])
	np.save(leadtxt+'lin.npy',image_params[5])
	#Save the edges
	with open(leadtxt+'edges', 'wb') as outfile:
	   pickle.dump(image_params[6], outfile, pickle.HIGHEST_PROTOCOL)
	print('done'+ leadtxt)

print('runs complete')
