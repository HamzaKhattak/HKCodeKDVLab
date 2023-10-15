# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 15:42:14 2023

@author: WORKSTATION
"""

  
'''
This tool takes images at a given framerate adapted from the multi camera example
on the Basler pypylon Github site
Currently set up for only two cameras.
'''

import os
from pypylon import genicam
from pypylon import pylon
import sys
import numpy as np
import tifffile as tfile
import time
class BCamCap:
	def __init__(self,whichcamsernum,secperframe):
		try:
			self.spf = secperframe
			# Get the transport layer factory.
			tlFactory = pylon.TlFactory.GetInstance()
		
			# Get all attached devices and exit application if no device is found.
			devices = tlFactory.EnumerateDevices()

			if len(devices) == 0:
				raise pylon.RUNTIME_EXCEPTION("No camera present.")
			sernum=[None]*len(devices)
			for i, d in enumerate(devices):
				sernum[i] = str(d.GetSerialNumber())
			whichcam = sernum.index(whichcamsernum)
			# Create instant camera for the rlevant axis
			self.cam = pylon.InstantCamera(tlFactory.CreateDevice(devices[whichcam]))
			
		
			# Create and attach all Pylon Devices.
			self.cam.Attach(tlFactory.CreateDevice(devices[whichcam]))
			self.cam.Open()
			self.cam.AcquisitionFrameRateEnable.SetValue(True)
			self.cam.AcquisitionFrameRate.SetValue(1/secperframe)
			self.cam.OffsetX.SetValue(0)
			self.cam.OffsetY.SetValue(0)
			self.dimarr = [self.cam.Height.GetMax(),self.cam.Width.GetMax()]
			self.sernum=self.cam.GetDeviceInfo().GetSerialNumber()
			self.cam.ExposureAuto.SetValue('Off')
			self.cam.GainAuto.SetValue('Off')
			self.cam.Close()
		except genicam.GenericException as e:
			# Error handling
			print("An exception occurred.", e.GetDescription())		
			
			
	def grabSequence(self,countOfImagesToGrab,fileName):
		
		
		# Maybe work in exit codes?
		# self.exitCode = 0
		#500 second maximum between frames
		tosave = fileName + '.tif' #filename to save
		
		try:
			times = np.zeros(countOfImagesToGrab)
			t0 = time.perf_counter()
			self.cam.StartGrabbing()
			grabResult=self.cam.RetrieveResult(500000, pylon.TimeoutHandling_ThrowException)
			imtosave=np.array([grabResult.GetArray()],dtype='uint8')
			
			mdat={"StartTime" : time.localtime(time.time()),
				   "IntendedDimensions": {
									    "time": countOfImagesToGrab,				
										  },
				   "Interval_ms": self.spf*1000
				 }
			
			tfile.imwrite(tosave,imtosave,append=True,bigtiff=True,metadata=mdat)
			
			# Grab c_countOfImagesToGrab from the cameras.
			for i in range(1,countOfImagesToGrab):
				if not self.cam.IsGrabbing():
					break
				#Grab images from cameras (could use the iterative method if more cameras end up being needed)
				grabResult=self.cam.RetrieveResult(500000, pylon.TimeoutHandling_ThrowException)
				times[i] = time.perf_counter()-t0
				#Save by appending, this is slower than just saving using the Pylon built in but is more convenient
				imtosave=np.array([grabResult.GetArray()],dtype='uint8')
				tfile.imwrite(tosave,imtosave,append=True,bigtiff=True,metadata=None)

		except genicam.GenericException as e:
			# Error handling
			print("An exception occurred.", e.GetDescription())
			self.cam.StopGrabbing()
		finally:
			self.cam.StopGrabbing()
			self.cam.Close()

	def grabFastSequence(self,countOfImagesToGrab,fileName):
		
		#Saves to RAM, good for fast short sequences
		# Maybe work in exit codes?
		# self.exitCode = 0
		#500 second maximum between frames
		tosave = fileName + '.ome.tif' #filename to save
		try:
			datshape = [countOfImagesToGrab,1,2048,2592] #hard coded in size, fix later, giving [0,32] for some reason
			imtosave=np.zeros(datshape,dtype='uint8')
			mdat={"StartTime" : time.localtime(time.time()),
				   "IntendedDimensions": {
									    "time": countOfImagesToGrab
										  },
				   "Interval_ms": self.spf*1000
				 }
			self.cam.StartGrabbing()
			# Grab c_countOfImagesToGrab from the cameras.
			for i in range(0,countOfImagesToGrab):
				if not self.cam.IsGrabbing():
					break
				#Grab images from camera (could use the iterative method if more cameras end up being needed)
				grabResult=self.cam.RetrieveResult(500000, pylon.TimeoutHandling_ThrowException)
				#Save by appending, this is slower than just saving using the Pylon built in but is more convenient
				imtosave[i]=np.array([grabResult.GetArray()],dtype='uint8')
				
			tfile.imwrite(tosave,imtosave,append=True,bigtiff=True,metadata=mdat)
		except genicam.GenericException as e:
			# Error handling
			print("An exception occurred.", e.GetDescription())
			self.cam.StopGrabbing()
		finally:
			self.cam.StopGrabbing()
			self.cam.Close()