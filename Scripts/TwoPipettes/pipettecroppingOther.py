# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 11:52:50 2022

@author: WORKSTATION
"""

import numpy as np
import matplotlib.pyplot as plt
import pathlib 
import xmltodict
from pyometiff import OMETIFFReader
import glob
import pims
import imageio 
import ndtiff as ndt

#Get the file names and create labels for the time steps and crop locations
filenames=glob.glob("*.avi")
#filenames=glob.glob("*/")
labels = ['File name','Time Step (s)','TopCrop x','TopCrop y','BotCrop x','BotCrop y','Split y']
imageinfos = [labels]

#Loop through each of the tiff files in the folder
for file in filenames:
	img_fpath = pathlib.Path(file)
	
	
	#for avi
	tstep = 0.05
	v = pims.PyAVReaderIndexed(str(img_fpath))
	left = v[0][:,:,0]
	right = v[-1][:,:,0]
	'''
	#infile = imageio.imread(str(img_fpath))
	infile= ndt.NDTiffDataset(file).as_array()[0,:,0,0]
	tstep = 0.25
	left = infile[0]
	right = infile[-1]
	'''
	#Side view cropping and selections
	fig = plt.figure('Pick top left and bottom right corner and then fit lines')
	plt.imshow(left,cmap='gray')
	plt.imshow(right,cmap='gray',alpha=0.5)
	plt.grid(which='both')
	print('Select crop points and then split line')
	crop_points = np.floor(plt.ginput(3,timeout=200)) #format is [[xmin,ymin],[xmax,ymax]]
	plt.close()
	imageinfo = [file,tstep] + crop_points[0].tolist() + crop_points[1].tolist() + [crop_points[2][1]]
	imageinfos = imageinfos + [imageinfo]



np.savetxt('runsparams.csv',imageinfos,delimiter=',',fmt='%s')



print('done')

