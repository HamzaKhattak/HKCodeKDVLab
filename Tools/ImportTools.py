from skimage.io import imread as imread2
from skimage.io import ImageCollection as ImageCollection2
import glob

def stackimport(FilePath):
	'''
	Just renaming the imread from skimage to import an image stack
	preferred method, need to include the filename.tif
	'''
	return imread2(FilePath)

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

