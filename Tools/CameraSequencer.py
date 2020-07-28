  
'''
This tool takes images at a given framerate adapted from the multi camera example
on the Basler pypylon Github site
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
		
		tosave = fileName + '.ome.tif' #filename to save
		
		try:
			atime=time.time()
			self.cameras.StartGrabbing()
			# Grab c_countOfImagesToGrab from the cameras.
			print(time.time()-atime)
			for i in range(countOfImagesToGrab*self.l):
				if not self.cameras.IsGrabbing():
					break
				grabResult=self.cameras.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
				
				
				tfile.imwrite(tosave,grabResult.GetArray(), append=True,bigtiff=True)
				
				'''
				grabResult1 = cameras[0].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
				btime=time.time()
				grabResult2 = cameras[1].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
				ctime=time.time()
		
				ctime=time.time()
				
				#tfile.imwrite('temptesting.tif',grabResult1.GetArray(), append=True)
				#tfile.imwrite('temptesting.tif',grabResult2.GetArray(), append=True)
				tfile.imwrite('temptesting2.ome.tif',np.array([grabResult1.GetArray(),grabResult2.GetArray()]), append=True,bigtiff=True)
				dtime=time.time()
				
				print(dtime-ctime)
				'''
			#cameras.StopGrabbing()
		except genicam.GenericException as e:
			# Error handling
			print("An exception occurred.", e.GetDescription())
			self.cameras.StopGrabbing()
		finally:
			self.cameras.StopGrabbing()
			self.cameras.Close()