import os, glob, pickle, re
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

'''
The next few bits of code are to get information on folders
'''

def split_on_letter(s):
	'''This code splits at the first letter to allow for sorting based on
	the first letter backslash W is for the nonaplhanumeric and backslash d for
	decimals. The hat then inverts that'''
	match = re.compile("[^\W\d]").search(s)
	return [s[:match.start()], s[match.start():]]

def split_at(s, c = '_', n = 1):
	'''
	This function can split at an underscore, dash etc depending on the input
	By default takes the first underscore
	'''
	words = s.split(c)
	return c.join(words[:n]), c.join(words[n:])

def stringtonum(s):
	'''
	Converts a string where leadering zeros are used to indicate decimals to a 
	float (ie 01 becomes 0.1 and 1 stays 1.0)
	'''
	leadzeros = len(s) - len(s.lstrip('0'))
	leadzeros=float(leadzeros)
	unscaledval = float(s)
	val = unscaledval*(10**(-leadzeros))
	return val

def namevelfind(s,splitfunction = split_on_letter, splitparams=[],numLoc = 0,):
	'''
	Extracts the speed from the file name based on how it is set up
	Super sketchy but works for now
	'''
	allstrings = splitfunction(s,*splitparams)
	rawnumstring=allstrings[numLoc]
	return stringtonum(rawnumstring)


def foldergen(mainfolderloc,splitfunc = namevelfind,sparams=[]):
	'''
	This function returns a sorted list of the folders in the current working directory
	The first argument
	mainfolderloc is where the images from the run are
	'''
	folderpaths=glob.glob(mainfolderloc+'/*/')
	foldernames=next(os.walk(mainfolderloc))[1]
	#filenames=glob.glob("*.tif") #If using single files
	
	#Empty list for the position vs velocity information or other info
	eList=[None]*len(folderpaths)
	#Sort the folders by the leading numbers
	velocitylist1=[splitfunc(i,*sparams) for i in foldernames]

	foldernames = [x for _,x in sorted(zip(velocitylist1,foldernames))]
	folderpaths = [x for _,x in sorted(zip(velocitylist1,folderpaths))]
	return folderpaths, foldernames, eList