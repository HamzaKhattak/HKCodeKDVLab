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
#%%

filenames=glob.glob("*.tif")
labels = ['File name','Time Step (s)','TopCrop x','TopCrop y','BotCrop x','BotCrop y','Split y']
imageinfos = [labels]

for file in filenames:
	img_fpath = pathlib.Path(file)
	
	reader = OMETIFFReader(fpath=img_fpath)
	img_array, metadata, xml_metadata = reader.read()
	image_metadata = xmltodict.parse(xml_metadata)['OME']['Image']["Pixels"]
	time_increment =float(image_metadata['@TimeIncrement'])
	time_increment_unit =image_metadata['@TimeIncrementUnit']
	
	if time_increment_unit =='ms':
		tstep = time_increment/1000
	elif time_increment_unit == 's':
		tstep = time_increment
	
	#Side view cropping and selections
	fig = plt.figure('Pick top left and bottom right corner and then fit lines')
	plt.imshow(img_array[0],cmap='gray')
	plt.imshow(img_array[-1],cmap='gray',alpha=0.5)
	plt.grid(which='both')
	print('Select crop points and then split line')
	crop_points = np.floor(plt.ginput(3,timeout=200)) #format is [[xmin,ymin],[xmax,ymax]]
	plt.close()
	imageinfo = [file,tstep] + crop_points[0].tolist() + crop_points[1].tolist() + [crop_points[2][1]]
	imageinfos = imageinfos + [imageinfo]
#%%


np.savetxt('runsparams.csv',imageinfos,delimiter=',',fmt='%s')
#%%
print(img_array[0].shape)
#%%

print('done')

