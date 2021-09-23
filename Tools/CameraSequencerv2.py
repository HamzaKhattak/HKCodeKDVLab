  
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
	'''
	modified from the Basler github grabmultiplecameras
	'''
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
				cam.OffsetX.SetValue(0)
				cam.OffsetY.SetValue(0)
				self.dimarr[i] = [cam.Height.GetMax(),cam.Width.GetMax()]
				self.sernum[i] = cam.GetDeviceInfo().GetSerialNumber()
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
		#Assumes all cameras have same dimensions, could be changed but would be slower
		tosave = fileName + '.ome.tif' #filename to save
		
		try:

			self.cameras.StartGrabbing()
			if self.l==1:
				imobj = self.cameras.RetrieveResult(500000, pylon.TimeoutHandling_ThrowException)			
				mdat={"StartTime" : time.localtime(time.time()),
					   "IntendedDimensions": {
										    "time": countOfImagesToGrab,
										    "position": 1,
										    "z": 1,
										    "channel": self.l
											  },
					   "Interval_ms": self.spf*1000
					 }
				#Check if converting to numpy array is required
				tfile.imwrite(tosave,imobj.GetArray(),append=True,bigtiff=True,metadata=mdat)
				# Grab c_countOfImagesToGrab from the cameras.
				for i in range(1,countOfImagesToGrab):
					if not self.cameras.IsGrabbing():
						break
					#Grab images from cameras (could use the iterative method if more cameras end up being needed)
					imobj = self.cameras.RetrieveResult(500000, pylon.TimeoutHandling_ThrowException)
					tfile.imwrite(tosave,imobj.GetArray(),append=True,bigtiff=True)


			else:
				scratcharray = np.zeros([self.l,self.dimarr[0][0],self.dimarr[0][1]],dtype='uint8')
				for viewnum in range(self.l)
					imobj = self.cameras.RetrieveResult(500000, pylon.TimeoutHandling_ThrowException)
					scratcharray[viewnum] = imobj.GetArray()
				
				mdat={"StartTime" : time.localtime(time.time()),
					   "IntendedDimensions": {
										    "time": countOfImagesToGrab,
										    "position": 1,
										    "z": 1,
										    "channel": self.l
											  },
					   "Interval_ms": self.spf*1000
					 }
				
				tfile.imwrite(tosave,scratcharray,planarconfig='separate',append=True,bigtiff=True,metadata=mdat)
				# Grab c_countOfImagesToGrab from the cameras.
				for i in range(1,countOfImagesToGrab):
					if not self.cameras.IsGrabbing():
						break
					#Grab images from cameras (could use the iterative method if more cameras end up being needed)
					for viewnum in range(self.l)
						imobj = self.cameras.RetrieveResult(500000, pylon.TimeoutHandling_ThrowException)
						scratcharray[viewnum] = imobj.GetArray()
						tfile.imwrite(tosave,scratcharray,planarconfig='separate',append=True,bigtiff=True)

		except genicam.GenericException as e:
			# Error handling
			print("An exception occurred.", e.GetDescription())
			self.cameras.StopGrabbing()
		finally:
			self.cameras.StopGrabbing()
			self.cameras.Close()

	def grabFastSequence(self,countOfImagesToGrab,fileName):
		
		#Saves to RAM, good for fast short sequences
		# Maybe work in exit codes?
		# self.exitCode = 0
		#500 second maximum between frames
		tosave = fileName + '.ome.tif' #filename to save
		try:
			#Single camera
			if self.l==1:
				datshape = [countOfImagesToGrab,self.dimarr[0][0],self.dimarr[0][1]] #dimension array issues
				imtosave=np.zeros(datshape,dtype='uint8')
				mdat={"StartTime" : time.localtime(time.time()),
					   "IntendedDimensions": {
										    "time": countOfImagesToGrab,
										    "position": 1,
										    "z": 1,
										    "channel": 1
											  },
					   "Interval_ms": self.spf*1000
					 }

				self.cameras.StartGrabbing()
				# Grab c_countOfImagesToGrab from the cameras.
				for i in range(0,countOfImagesToGrab):
					if not self.cameras.IsGrabbing():
						break
					#Grab images from cameras
					imobj = self.cameras.RetrieveResult(500000, pylon.TimeoutHandling_ThrowException)
					imtosave[i] = imobj.GetArray()
					
				tfile.imwrite(tosave,imtosave,append=True,bigtiff=True,metadata=mdat)
			#Multi camera
			else:
				datshape = [countOfImagesToGrab,self.l,self.dimarr[0][0],self.dimarr[0][1]] #hard coded in size, fix later, giving [0,32] for some reason
				imtosave=np.zeros(datshape,dtype='uint8')
				mdat={"StartTime" : time.localtime(time.time()),
					   "IntendedDimensions": {
										    "time": countOfImagesToGrab,
										    "position": 1,
										    "z": 1,
										    "channel": 2
											  },
					   "Interval_ms": self.spf*1000
					 }
				self.cameras.StartGrabbing()
				scratcharray = np.zeros([self.l,self.dimarr[0][0],self.dimarr[0][1]])

				# Grab c_countOfImagesToGrab from the cameras.
				for i in range(0,countOfImagesToGrab):
					if not self.cameras.IsGrabbing():
						break
					for j in range(self.l)
						#Grab images from cameras (could use the iterative method if more cameras end up being needed)
						imobj = self.cameras.RetrieveResult(500000, pylon.TimeoutHandling_ThrowException)
						scratcharray[j] = imobj.GetArray()
					#Save by writing to array, this may be slower than just saving using the Pylon built in but is more convenient
					imtosave[i] = scratcharray
					
				tfile.imwrite(tosave,imtosave,planarconfig='separate',append=True,bigtiff=True,metadata=mdat)
		except genicam.GenericException as e:
			# Error handling
			print("An exception occurred.", e.GetDescription())
			self.cameras.StopGrabbing()
		finally:
			self.cameras.StopGrabbing()
			self.cameras.Close()