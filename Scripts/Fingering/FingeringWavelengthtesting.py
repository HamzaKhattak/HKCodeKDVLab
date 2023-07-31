# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:06:15 2023

@author: hamza
"""

import numpy as np
import imageio as io
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2
from scipy import signal
from scipy.optimize import curve_fit
#%%
baseim = io.imread('baseim.tif')
staticim = io.imread('staticunwashed.tif')
movingim = io.imread('unwashedmove.tif')


#%%
#leveling image


def radial_profile(data, center):
	'''
	Calculates the radial distribution function for an image for a given center
	'''
	y, x = np.indices((data.shape))
	r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
	r = r.astype(int)

	tbin = np.bincount(r.ravel(), data.ravel())
	nr = np.bincount(r.ravel())
	radialprofile = tbin / nr
	return radialprofile 

def fitfunc(x,a,b,c):
	return a*x**2+b*x+c

def profilefit(radialdist,function):
	'''
	Fits to the radial distrubution function
	'''
	x=np.arange(len(radialdist),dtype=int)
	y = radialdist
	popt, perr = curve_fit(fitfunc,x, y)
	return x,y,function, popt, perr
	

def subtractfit(data,center,function,fitparams):
	'''
	Substracts off a radial distribution fit from an image
	'''
	data = data.astype(int)
	y1,x1 = np.indices((data.shape))
	r1 = np.sqrt((x1 - center[0])**2 + (y1 - center[1])**2)
	return data-function(r1,*fitparams)

def rescale(data):
	'''

	rescale data to uint8

	'''
	newdat = 255*(data-np.min(data))/(np.max(data)-np.min(data))
	return newdat.astype(np.uint16)

center = [1383,965]
brightnessprofile = radial_profile(baseim, [1383,965])
fits = profilefit(brightnessprofile,fitfunc)
plt.figure()
plt.plot(fits[0],fits[1])
plt.plot(fits[0],fits[2](fits[0],*fits[3]))

inputimages=[baseim,staticim,movingim]
correctedimages=[None]*3

for i in range(3):
	correctedimages[i]=subtractfit(inputimages[i],center,fitfunc,fits[3])
	correctedimages[i]=rescale(correctedimages[i])






#%%

def fftfinder(inputimage,boxdeletesize = 6):
	'''
	Calculate 2d Fourier transform and mask center DC region
	'''
	th1 = np.fft.fftshift(np.fft.fft2(inputimage))
	DCdeletey = [int(th1.shape[0]/2-boxdeletesize),int(th1.shape[0]/2+boxdeletesize)]
	DCdeletex = [int(th1.shape[1]/2-boxdeletesize),int(th1.shape[1]/2+boxdeletesize)]
	th1[DCdeletey[0]:DCdeletey[1],DCdeletex[0]:DCdeletex[1]]=0.1
	return th1

def autocorrelatefinder(inputimage,boxdeletesize = 6):
	'''
	Calculate 2d Fourier transform and mask center DC region
	'''
	th1 = signal.correlate(inputimage-np.mean(inputimage),inputimage,mode='full')
	DCdeletey = [int(th1.shape[0]/2-boxdeletesize),int(th1.shape[0]/2+boxdeletesize)]
	DCdeletex = [int(th1.shape[1]/2-boxdeletesize),int(th1.shape[1]/2+boxdeletesize)]
	th1[DCdeletey[0]:DCdeletey[1],DCdeletex[0]:DCdeletex[1]]=0.1
	return th1

def centerzoomer(inputarray,size):
	'''
	Zoom in on the center of an image for a given size [x,y]
	'''
	y0 , x0 = [int(x/2) for x in [inputarray.shape[0],inputarray.shape[1]]]
	diffx=int(size[0])
	diffy=int(size[1])
	x1 , x2 , y1, y2 = [x0-diffx,x0+diffx,y0-diffy,y0+diffy]
	return inputarray[y1:y2,x1:x2]


for i in range(3):
	blurred = cv2.GaussianBlur(correctedimages[i],(3,3),0)
	fts[i] = fftfinder(blurred)


#%%
from matplotlib import colors
c_white = colors.colorConverter.to_rgba('red',alpha = 0)
c_red= colors.colorConverter.to_rgba('red',alpha = .3)
cmap_rb = colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white,c_red],512)

'''
Finding the contact patch using thresholds
'''

from scipy import ndimage
def thresholdfinder(inputimage):
	start = inputimage
	blurred = cv2.GaussianBlur(start,(3,3),0)
	ret, thresh1 = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)
	thresh2 = ndimage.binary_closing(thresh1, structure=np.ones((11,11)))
	return thresh2


plt.imshow(correctedimages[2],cmap='gray')
plt.imshow(thresholdfinder(correctedimages[2]),cmap=cmap_rb)

#%%

'''
Extract main contact patch
'''
def largestpatchfind(inputTFarray):
	labeled, nr_objects = ndimage.label(inputTFarray)
	objectareas = np.bincount(labeled.ravel())[1:] #0 is the empty
	indexofmax = np.argmax(objectareas)+1
	return labeled==indexofmax

mainpatch = largestpatchfind(thresholdfinder(correctedimages[2]))
#plt.imshow(mainpatch)



def tightcropper(inputmainpatch,imreturn = True):

	hor_minlocs = np.argmax(inputmainpatch,axis=1)
	hor_minlocs2 = np.argmax(inputmainpatch[:,::-1],axis=1)
	vert_minlocs = np.argmax(inputmainpatch,axis=0)
	vert_minlocs2 = np.argmax(inputmainpatch[::-1],axis=0)
	
	maxfindingarrays = [hor_minlocs,hor_minlocs2,vert_minlocs,vert_minlocs2] 
	extents = [None]*4
	for i in range(4):
		a = maxfindingarrays[i]
		extents[i] = np.min(a[np.nonzero(a)])
	extents[1]=-extents[1]
	extents[3]=-extents[3]
	tightcrop = inputmainpatch[extents[2]:extents[3],extents[0]:extents[1]]
	if imreturn == False:
		tightcrop = extents
	return tightcrop

mainpatches = [tightcropper(
	largestpatchfind(
	thresholdfinder(
		correctedimages[i+1]))) for i in range(2)]

plt.imshow(mainpatches[1])
#%%
from skimage.morphology import convex_hull_image
'''
Do the FFt on convex hull of the actual image rather than the thresholding
'''
def convexhullfind(inputim):
	mainpatch = largestpatchfind(
	thresholdfinder(
		inputim))
	#test = np.ma.masked_array(x, mask=[0, 0, 0, 1, 0])
	boundingsphere = convex_hull_image(mainpatch)
	#plt.imshow(boundingsphere)
	extents = tightcropper(boundingsphere,False)
	
	boundingsphere=boundingsphere[extents[2]:extents[3],extents[0]:extents[1]]
	cropped=inputim[extents[2]:extents[3],extents[0]:extents[1]]
	
	maskedpatch = cropped*boundingsphere
	return maskedpatch


def rextentfind(inputpolar):
	'''
	Finds the extent in the radial dimension of a radial polar image of fingering
	Works since fingering is where brigthness increase starts
	Takes radial image as input, should be cropped to region of interest in theta
	'''
	variancedat = np.var(inputpolar,axis=0) 
	variancedat = variancedat-np.mean(variancedat[:400])
	variancedat = variancedat/np.max(variancedat)
	smoothedvariance = signal.savgol_filter(variancedat,15,3)
	#find the peaks and take the first one, pretty manual right now
	#Generally very little variance until the fingering
	peaks, plateau = signal.find_peaks(smoothedvariance, height=.01)
	leftend = peaks[np.where(peaks>200)][0] #Get the first peak not in the center
	#To get the right end flip the image and choose value at shortest span
	flipped = inputpolar[:,::-1]
	flnonzeros = np.argmax(flipped>0,axis=1) #Get values for first non-zero elements
	xlen = flipped.shape[1]
	rightend = xlen - np.max(flnonzeros)
	#plt.plot(flnonzeros)

	return [leftend,rightend]


def leadtrailfingerfind(inputpolar,halfangle):
	'''
	Parameters
	----------
	inputpolar : polar coordinate image of the fingering, assumes 0 degrees is front and that it loops
	halfangle: value in degrees that decides how much of the radial values will be scanned

	Returns
	-------
	A list containing
	-Cropped images of the fingers as well as how far they start from the center
	'''
	
	anglenum = inputpolar.shape[1] #number of angle values in the image to calculate cutoff ints
	cutoffs = int(halfangle*anglenum/360)
	centervalue = int(anglenum/2)
	
	#Get the crop of the left values
	leftcrop=polar[centervalue-cutoffs:centervalue+cutoffs]
	startl, endl  = rextentfind(leftcrop)
	leftcrop=leftcrop[:,startl:endl]
	
	#Get a crop of the right values, same method but need numpy.take to wrap around
	indicesfront = range(i-cutoffs,i+cutoffs)
	rightcrop=polar.take(indicesfront, axis=0,mode='wrap')
	startr, endr = rextentfind(rightcrop)
	rightcrop = rightcrop[:,startr:endr]
	return [startl,leftcrop],[startr,rightcrop]


def xgenerator(initial,inputimage,thetpixsize):
	'''
	Generates the x arrays for the cross correlation (correcting for r)
	initial is the starting r in pixels
	theta pix size is the size of a theta pixel in radians
	shift is in theta pixels
	
	***Fix the shift values in this code**
	'''
	
	thetalen = inputimage.shape[0]
	rlen = inputimage.shape[1]
	
	#cross correlation goes from negative to positive
	x0 = np.linspace(-thetalen+1,thetalen-1,num=2*thetalen-1)
	x0 = x0*thetpixsize #convert from theta to r
	#radius needs to be added
	rvals = np.linspace(initial,initial+rlen-1,num = rlen)
	
	xall = np.outer(rvals,x0)

	return xall


'''
Completing the cross correlation sample and sample of how to get it back to real lengths
'''

from scipy import interpolate
def interpolatorcorr(inputimage,xmatrix):
	'''
	Take an input cropped radial image and matrix of x positions and converts
	it it a series of autocorrelation interpolation functions
	'''
	#shift each line by the mean to get rid of DC shift
	linemeans = np.mean(inputimage,axis=0)
	shiftedim = np.subtract(inputimage,linemeans)
	
	allinterps = [None]*inputimage.shape[1] #Create list to store interp functions
	
	for i in range(inputimage.shape[1]):
		x = xmatrix[i]
		y = signal.correlate(shiftedim[:,i],shiftedim[:,i])
		y = y/np.max(y)
		allinterps[i] = interpolate.interp1d(x, y, kind='linear', axis=-1, assume_sorted=True)
		
	return allinterps

def interpolationaverager(interps,avrange,numvals):
	'''
	Averages a series of itnerpolation functions
	'''
	x_all = np.linspace(avrange[0], avrange[1], num=numvals)
	ys = np.array([interps[i](x_all) for i in range(len(interps))])
	combointer = np.mean(np.vstack(ys),axis=0)
	combointer = combointer-np.mean(combointer)
	return np.array([x_all,combointer])

from scipy.fft import fft, fftfreq
def FFTcorr(inputautocorr):
	
	#N = SAMPLE_RATE * DURATION
	#xf = fftfreq(N, 1 / SAMPLE_RATE)
	N = len(inputautocorr[0])
	T = np.abs(inputautocorr[0,1]-inputautocorr[0,0])
	yf = fft(inputautocorr[1])
	ftransform = 2.0/N * np.abs(yf[1:N//2])
	xf = fftfreq(N, T)[1:N//2]
	wavelength = 1/xf
	return [wavelength,ftransform]


imageofinterest = correctedimages[1]
maincenter = [988,1384]

maskedpatch = convexhullfind(imageofinterest)
halfangleuse = 35
plt.imshow(imageofinterest)
#%%
plt.figure()
plt.imshow(maskedpatch)
#center = [842,688]
center = [623,635]
#%%
#Get a polar image
polar=cv2.linearPolar(maskedpatch,center,900,cv2.WARP_FILL_OUTLIERS)
plt.figure()
plt.imshow(polar)

#Crop the images to the half angle and r extents
croppedradialims = leadtrailfingerfind(polar, halfangleuse)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(croppedradialims[0][1])
ax2.imshow(croppedradialims[1][1])

#Calculate the cross correlation along each axis
thetapix = 2*np.pi/polar.shape[0]
xmatleft = xgenerator(croppedradialims[0][0],croppedradialims[0][1],thetapix)
xmatright = xgenerator(croppedradialims[1][0],croppedradialims[1][1],thetapix)
allinterpsleft = interpolatorcorr(croppedradialims[0][1],xmatleft)
allinterpsright = interpolatorcorr(croppedradialims[1][1],xmatright)






#Calculate the average cross correlation
averagecorrleft = interpolationaverager(allinterpsleft,[-np.max(xmatleft[0]),np.max(xmatleft[0])],len(xmatleft[0]))
averagecorrright = interpolationaverager(allinterpsright,[-np.max(xmatright[0]),np.max(xmatright[0])],len(xmatright[0]))

plt.figure()
plt.plot(averagecorrleft[0],averagecorrleft[1],label='left')
plt.plot(averagecorrright[0],averagecorrright[1],label='right')
plt.xlabel('shift (pixels)')
plt.ylabel('intensity')
plt.legend()

#Get a FFT of the cross correlation to get a wavelength associated with the feature
rcorrfft = FFTcorr(averagecorrright)
lcorrfft = FFTcorr(averagecorrleft)
plt.figure()
plt.plot(rcorrfft[0],rcorrfft[1],label = 'right')
plt.xlabel('wavelength (pixels)')
plt.plot(lcorrfft[0],lcorrfft[1],label = 'left')
plt.ylabel('intensity')
plt.legend()
