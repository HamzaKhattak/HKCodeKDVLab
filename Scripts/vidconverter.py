# -*- coding: utf-8 -*-
'''
This code is for quickly combining files for later analysis and then saving a video to make for easy playing
'''

import os, glob, imageio
#need to install imageio and imagio-ffmpeg and tkinter 
import tifffile as tf
import numpy as np
import tkinter as tk
from tkinter import filedialog
from skimage.io import ImageCollection
import ndtiff as ndt

root = tk.Tk()
root.withdraw()

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
	numFrames = len(tifobj.pages)
	return tf.imread(FilePath,key=slice(0,numFrames))



#%%


dualim = input('Type 1 for a single angle and 2 for two angles beside each other (Exact same length/size) \n')
collect = input('Type 1 for a single input file and 2 for a collection (collection is only for 1 view) \n')

if dualim == '1':
	file_path = filedialog.askopenfilename()
	outFile = input('Output file name (include .mp4 extension) \n')
	inFPS = input('FPS (outputs at 30fps for device compatibility so will cut/add frames as needed) \n')
	#Import the tif files in a folder
	#imageframes=ito.omestackimport(dataDR)
	if collect == '1':
		imageframes=fullseqimport(os.path.join(file_path))
	if collect == '2':
		imfilenames=sorted(glob.glob(os.path.dirname(file_path) + "/*.tif"))
		imageframes = fullseqimport(os.path.join(imfilenames[0]))
		for i in range(len(imfilenames)-1):
			temp = fullseqimport(os.path.join(imfilenames[i+1]))
			imageframes = np.append(imageframes,temp,axis=0)
	if collect == '3':
		impath=os.path.dirname(file_path)
		ims = ndt.NDTiffDataset(impath)
		imageframes = ims.as_array()[0,:,0,0]

	
	
	imageio.mimwrite(outFile, imageframes , input_params=['-r',inFPS],  output_params=['-r', '30'],  ffmpeg_params=[
	                                "-vcodec",
	                                "libx264",
	                                "-crf",
	                                "20",])
elif dualim == '2':
	file_path1 = filedialog.askopenfilename()
	file_path2 = filedialog.askopenfilename()
	outFile = input('Output file name (include .mp4 extension) \n')
	inFPS = input('FPS (outputs at 30fps for device compatibility so will cut/add frames as needed) \n')
	imageframes1=fullseqimport(os.path.join(file_path1))
	imageframes2=fullseqimport(os.path.join(file_path2))
	combo = np.concatenate((imageframes1, imageframes2), axis=1)
	imageio.mimwrite(outFile, combo , input_params=['-r',inFPS],  output_params=['-r', '30'],  ffmpeg_params=[
	                                "-vcodec",
	                                "libx264",
	                                "-crf",
	                                "20",] )
else:
	print('invalid input')

#ffmpeg -i input.mkv -r 16 -filter:v "setpts=0.25*PTS" output.mkv