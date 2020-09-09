import os, glob, pickle, re
import numpy as np
from skimage.io import imread as imread2
from skimage.io import ImageCollection as ImageCollection2
import tifffile as tf



'''
Code to deal with file naming structure
'''

#First need to create the folders for saving
def foldercreate(runspeed):
	#Creates a unique folder in the working directory for a given runspeed in mm/s
	#Avoid periods and special characters in numbers so 0.1um/s becomes 0p1ums etc
	#When reading simply replace the p with a "."
	prefix=str(runspeed*1000)
	prefix=prefix.replace(".","p")
	prefix = prefix + "ums"
	#Ensure that folders are not overwritten
	i = 0
	while os.path.exists(prefix+str(i)):
		i+=1
		time.sleep(0.1)
	createdfolder = prefix+str(i)
	os.mkdir(createdfolder)
	return createdfolder

#Then need to be able to import from those folders
def numberreturn(numstr):
	'''
	Get the speed from a 10p4ums0 type string
	simple to change if naming convention changes
	'''
	intstr = numstr.split('u')[0]
	intstr = intstr.replace('p','.')
	speed = float(x)
	return speed


def folderlistgen(mainfolderloc,splitfunc = numberreturn,sparams=[]):
	'''
	This function returns a sorted list of the folders in the current working directory
	The first argument
	mainfolderloc is where the subfolders of each run from the experiment from the run are
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


'''
Code to import images series
'''
def preimport(FilePath):
	'''
	This function creates a tifffile object that can be referenced for image import operations
	Simply a renaming of the tifffile package to keep it seperate
	This object is more of a reference to the file and has info like number of pages etc
	'''
	return tf.TiffFile(FilePath)

def singlesliceimport(FilePath,ival):
	'''
	This imports only a single frame of a tiff sequence
	'''
	tifobj = preimport(FilePath)
	return tifobj.pages[ival].asarray()


def fullseqimport(FilePath):
	'''
	This object imports the entire sequence of tiff images
	'''
	tifobj = preimport(FilePath)
	numFrames = len(tifobj)
	return tf.imread(FilePath,key=slice(0,numFrames))

def getimpath(FolderPath):
	'''
	Return the first tif image path in a given folder
	'''
	return sorted(glob.glob(FolderPath + "/*.tif"))[0]

'''
Some generally useful tools
'''
def cropper(seq,x1,x2,y1,y2):
    if seq.ndim==2:
        return seq[y1:y2, x1:x2]
    else:
        return seq[:, y1:y2, x1:x2]

def cropper2(seq,croparray):
    x1, y1, x2, y2 = np.uint(croparray.flatten())
    if seq.ndim==2:
        return seq[y1:y2, x1:x2]
    else:
        return seq[:, y1:y2, x1:x2]


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
Older code kept in case used later
'''
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
	emptyList=[None]*len(folderpaths)
	#Sort the folders by the leading numbers
	velocitylist1=[splitfunc(i,*sparams) for i in foldernames]

	foldernames = [x for _,x in sorted(zip(velocitylist1,foldernames))]
	folderpaths = [x for _,x in sorted(zip(velocitylist1,folderpaths))]
	return folderpaths, foldernames, emptyList