  
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
import time
import tifffile as tfile

class BCamCap:
	def __init__(self,maxCamerasToUse,secperframe):
		try:
			self.spf = secperframe
			# Get the transport layer factory.
			tlFactory = pylon.TlFactory.GetInstance()
		
			# Get all attached devices and exit application if no device is found.
			devices = tlFactory.EnumerateDevices()
			if len(devices) == 0:
				raise pylon.RUNTIME_EXCEPTION("No camera present.")
		
			# Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
			self.cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))
		
			self.l = self.cameras.GetSize()
			
			self.dimarr=np.zeros((self.l,2),dtype='uint8')
			self.sernum=[None]*self.l
			# Create and attach all Pylon Devices.
			for i, cam in enumerate(self.cameras):
				#Some parameters I'm using now, may change in future.
				cam.Attach(tlFactory.CreateDevice(devices[i]))
				cam.Open()
				cam.AcquisitionFrameRateEnable.SetValue(True)
				cam.AcquisitionFrameRate.SetValue(1/secperframe)
				self.dimarr[i] = [cam.Height.GetValue(),cam.Width.GetValue()]
				self.sernum[i]=cam.GetDeviceInfo().GetSerialNumber()
				cam.ExposureAuto.SetValue('Off')
				cam.GainAuto.SetValue('Off')
				cam.Close()
		except genicam.GenericException as e:
			# Error handling
			print("An exception occurred.", e.GetDescription())		
			
			
	def grabSequence(self,countOfImagesToGrab,fileName):
		
		
		# Maybe work in exit codes?
		# self.exitCode = 0
		#500 second maximum between frames
		tosave = fileName + '.ome.tif' #filename to save
		
		try:
			self.cameras.StartGrabbing()
			grabResult1=self.cameras.RetrieveResult(500000, pylon.TimeoutHandling_ThrowException)
			grabResult2=self.cameras.RetrieveResult(500000, pylon.TimeoutHandling_ThrowException)
			imtosave=np.array([grabResult1.GetArray(),grabResult2.GetArray()],dtype='uint8')
			
			mdat={"StartTime" : time.localtime(time.time()),
				   "IntendedDimensions": {
									    "time": countOfImagesToGrab,
									    "position": 1,
									    "z": 1,
									    "channel": 2
										  },
				   "Interval_ms": self.spf*1000
				 }
			
			tfile.imwrite(tosave,imtosave,planarconfig='separate',append=True,bigtiff=True,metadata=mdat)
			# Grab c_countOfImagesToGrab from the cameras.
			for i in range(1,countOfImagesToGrab):
				if not self.cameras.IsGrabbing():
					break
				#Grab images from cameras (could use the iterative method if more cameras end up being needed)
				grabResult1=self.cameras.RetrieveResult(500000, pylon.TimeoutHandling_ThrowException)
				grabResult2=self.cameras.RetrieveResult(500000, pylon.TimeoutHandling_ThrowException)
				#Save by appending, this is slower than just saving using the Pylon built in but is more convenient
				imtosave=np.array([grabResult1.GetArray(),grabResult2.GetArray()],dtype='uint8')
				tfile.imwrite(tosave,imtosave,planarconfig='separate',append=True,bigtiff=True)

		except genicam.GenericException as e:
			# Error handling
			print("An exception occurred.", e.GetDescription())
			self.cameras.StopGrabbing()
		finally:
			self.cameras.StopGrabbing()
			self.cameras.Close()