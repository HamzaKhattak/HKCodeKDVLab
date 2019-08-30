import os, glob, pickle
import numpy as np
from skimage.io import imread as imread2
from skimage.io import ImageCollection as ImageCollection2

def stackimport(FilePath):
	'''
	Just renaming the imread from skimage to import an image stack
	preferred method, need to include the filename.tif
	'''
	return imread2(FilePath)
	
def omestackimport(FolderPath):
	'''
	Just renaming the imread from skimage to import an image stack
	preferred method, need to include the filename.tif of the first image
	'''
	imfilenames=sorted(glob.glob(FolderPath + "/*.tif"))
	return imread2(imfilenames[0])

def cropper(seq,x1,x2,y1,y2):
    if seq.ndim==2:
        return seq[y1:y2, x1:x2]
    else:
        return seq[:, y1:y2, x1:x2]

def folderimportdeprecated(FolderLoc,extension):
	'''

	'''
	imfilenames=glob.glob(FolderLoc + "/*" + extension)
	numFiles=len(imfilenames)
	testimage=imread2(imfilenames[0])
	dim=testimage.shape
	image_list=np.zeros([numFiles,dim[0],dim[1]])   
	for i in range(numFiles):
		image_list[i]=imread2(imfilenames)

def folderimport(FolderLoc,extension):
	'''
	Just renaming the imagecollection command
	extension is .png .jpeg etc, need to include the .
	'''
	col_dir=FolderLoc+'/*'+extension
	return ImageCollection2(col_dir)

def folderstackimport(FolderLoc):
	'''
	Imports stacks of non-ome tifs from a folder depending on if there are multiple
	Will return double if there are multiple ome tiffs
	'''
	imfilenames=sorted(glob.glob(FolderLoc + "/*.tif"))
	if len(imfilenames) == 1:
		mainimg = stackimport(imfilenames[0])

	else:
		mainimg=stackimport(imfilenames[0])
		for i in np.arange(len(imfilenames)-1):
			img=stackimport(imfilenames[i+1])
			mainimg=np.concatenate((mainimg,img))
	return mainimg


def savelistnp(filepath,data):
	'''
	Saves lists of numpy arrays using pickle so they don't become objects
	'''
	with open(filepath, 'wb') as outfile:
		   pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)

def openlistnp(filepath):
	'''
	Opens lists of numpy arrays using pickle
	'''
	with open(filepath, 'rb') as infile:
	    result = pickle.load(infile)
	return result