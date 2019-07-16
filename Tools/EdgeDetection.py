
'''
This module includes code that detects edges in an image as well
as code to crop images
'''

import skimage.morphology as morph
import scipy.ndimage.morphology as morph2
from skimage import feature
import numpy as np

def cropper(seq,x1,x2,y1,y2):
    if seq.ndim==2:
        return seq[y1:y2, x1:x2]
    else:
        return seq[:, y1:y2, x1:x2]

def edgedetector(inimage,background,threshval,obsSize,cannysigma):
	'''
	This function finds the edges of a cropped image of a pipette and droplet
	Image must be black on a brighter backdrop. Returns the result as a numpy
	array type object
	Arguments are:
	    inimage: The input image
	    background: A background image (or a 0 image)
	    threshval: Threshold to use for binary thresholding of image, may be negative
	    obSize: Maximum size of noise object to remove, choose smallest value that removes dust
	    cannysigma: Choose value of guassian blur for edge detection
	'''
	#Subtract background if needed and select image, droplet should be high so invert
	imsub=background-inimage

	#Create thresholded image
	threshimage=imsub>threshval

	oldtopbottom=threshimage[[0,-1],:]
	threshimage[[0,-1],:]=True
	#Add pixel border to top and bottom to make holes at extrema are filled
	#Fill holes
	threshimage=morph2.binary_fill_holes(threshimage)
	threshimage[[0,-1],:]=oldtopbottom
	#Remove specs
	threshimage=morph.remove_small_objects(threshimage,obsSize)
	#Find the edges
	edgedetect=feature.canny(threshimage, sigma=cannysigma)
	return edgedetect

def edgestoxy(edges):
	return np.flip(np.argwhere(edges),1)