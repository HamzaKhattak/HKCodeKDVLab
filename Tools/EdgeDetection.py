
'''
This module includes code that detects edges in an image as well
as code to crop images
'''

import skimage.morphology as morph
import scipy.ndimage.morphology as morph2
from skimage import feature

def cropper(seq,x1,x2,y1,y2,singleimage=False):
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

    
    threshimage=imsub>threshval
    #Fill holes
    threshimage=morph2.binary_fill_holes(threshimage)
    #Remove specs
    threshimage=morph.remove_small_objects(threshimage,obsSize)
    #Find the edges
    edgedetect=feature.canny(threshimage, sigma=cannysigma)
    return edgedetect