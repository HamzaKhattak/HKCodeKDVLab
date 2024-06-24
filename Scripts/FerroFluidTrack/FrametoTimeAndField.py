# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 19:56:15 2024

@author: hamza
"""

from pyometiff import OMETIFFReader
import numpy as np





import os, glob, imageio, re
#need to install imageio and imagio-ffmpeg and tkinter 
import tifffile as tf
import numpy as np
import ndtiff as ndt
from datetime import datetime
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


'''
This section of code simply imports the fields from the Gaussmeter capture code
and creates an interp1d function that allows you to get a field for a given time
'''

def findGuassVals(fields_path, ims_path,savename):
	
	fields = np.loadtxt(fields_path,delimiter=',') #gives field by epoch time
	
	fieldf = interp1d(fields[:,0], fields[:,1],fill_value="extrapolate") #Create interp
	
	
	
	'''
	This section gets the time points for the image using the ome tiff metadata
	Gets both the time points using just time increment as well as using the per frame
	time points
	'''
	
	#Use tifffile to import
	tifobj = tf.TiffFile(ims_path)
	numFrames = len(tifobj.pages)
	#imageframes =  tf.imread(file_path,key=slice(0,numFrames))
	
	metdat = tifobj.imagej_metadata #Basic metadata
	omexml = tifobj.ome_metadata #Omexml metadata
	
	#Get the main info metadata and find the start time
	mainmetdat = metdat['Info']

	starttime = re.findall('"StartTime": "(.+?)"', mainmetdat)[0]
	starttime = datetime.strptime(starttime,'%Y-%m-%d %H:%M:%S.%f %z')
	startepochtime = starttime.timestamp()
	
	
	#Find the time increment from the xml metadata

	keywordiu = 'TimeIncrementUnit='
	match2 = re.findall(f'{keywordiu}"(.+?)" ', omexml)[0]
	
	'''
	
	#Getting times with the time increment method
	#make sure correct units
	keyword1 = 'TimeIncrement='
	match1 = re.findall(f"{keyword1}.*?(\d+[.]\d+)", omexml)[0]
	if match2=='ms':
		timestep = float(match1)/1000
	if match2 == 's':
		timestep = float(match1)
	
	#in case there is an error in the metadata, fall back on manual timestep
	inFPS = str(1/timestep)
	
	byincrementtimes = np.linspace(startepochtime,startepochtime+timestep*numFrames,num=numFrames)
	'''
	
	#Section the gets the time points using per frame time captures
	#find the keyword that gives each time point
	keyword = 'DeltaT='
	match = re.findall(f"{keyword}.*?(\d+[.]\d+)", omexml)
	#Turn into numpy array and add the initial epochtime
	
	if match2=='ms':
		factor = 1/1000
	if match2 == 's':
		factor = 1
	
	perframetimes = np.array(match,dtype=float)*factor+startepochtime
	
	'''
	This section of code uses the above and outputs an array giving:
	times since start, epoch times, and field strength in gauss for the experiment
	Uses the perframetimes since likely a bit more accurate
	'''
	
	fieldsarray = fieldf(perframetimes)
	start0time = perframetimes-perframetimes[0]
	tosave = np.transpose([perframetimes,start0time,fieldsarray])	
	np.savetxt(savename+'.csv',tosave,delimiter=',')
	return tosave
