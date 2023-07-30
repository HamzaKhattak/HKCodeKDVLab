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
#%%
baseim = io.imread('baseim.tif')
staticim = io.imread('staticunwashed.tif')
movingim = io.imread('unwashedmove.tif')


#%%
#leveling image

from scipy.optimize import curve_fit

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

plt.figure()
fig, ax = plt.subplots(1, 3)
for i in range(3):
	ax[i].imshow(correctedimages[i],cmap='gray')
#%%
plt.plot(radial_profile(correctedimages[1], center))



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
	th1 = signal.correlate(inputimage-np.mean(maskedpatch),inputimage,mode='full')
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

fts=[None]*3
for i in range(3):
	blurred = cv2.GaussianBlur(correctedimages[i],(3,3),0)
	fts[i] = fftfinder(blurred)


fig, ax = plt.subplots(1, 3)
for i in range(3):
	ax[i].imshow(abs(centerzoomer(fts[i],[100,100])),cmap='gray')
#%%

#%%

'''
Using find peaks to maybe reconstruct some part of the image
'''
newim = correctedimages[1]

x = newim[1000]

peaks, _ = signal.find_peaks(x, prominence=1,width=5,distance=10)
plt.plot(peaks, x[peaks]+1000, "xr"); plt.plot(x+1000); plt.legend(['distance'])
#%%


#%%
from matplotlib import colors
c_white = colors.colorConverter.to_rgba('red',alpha = 0)
c_red= colors.colorConverter.to_rgba('red',alpha = .3)
cmap_rb = colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white,c_red],512)

'''
Finding the contact patch using thresholds
'''

import scipy.ndimage as snd
def thresholdfinder(inputimage):
	start = inputimage
	blurred = cv2.GaussianBlur(start,(3,3),0)
	ret, thresh1 = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)
	thresh2 = snd.binary_closing(thresh1, structure=np.ones((11,11)))
	return thresh2


plt.imshow(correctedimages[2],cmap='gray')
plt.imshow(thresholdfinder(correctedimages[2]),cmap=cmap_rb)

#%%
from scipy import ndimage
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
mainpatch1 = largestpatchfind(
thresholdfinder(
	correctedimages[2]))
#test = np.ma.masked_array(x, mask=[0, 0, 0, 1, 0])
boundingsphere = convex_hull_image(mainpatch1)
#plt.imshow(boundingsphere)
extents = tightcropper(boundingsphere,False)

boundingsphere=boundingsphere[extents[2]:extents[3],extents[0]:extents[1]]
cropped=correctedimages[2][extents[2]:extents[3],extents[0]:extents[1]]

maskedpatch = cropped*boundingsphere
plt.imshow(maskedpatch)
#%%
'''
Get fourier of above
'''
tft = fftfinder(maskedpatch)
plt.imshow(plt.imshow(np.abs(centerzoomer(tft,[100,100]))))
#%%
directionalfft1 = np.fft.fft(maskedpatch,axis=0)
directionalfft2 = np.fft.fft(maskedpatch,axis=1)
plt.plot(np.mean((abs(directionalfft1)),axis=1)[5:-5])
plt.plot(np.mean((abs(directionalfft2)),axis=0)[5:-5])
#%%

print(maskedpatch.shape)
mean1 = np.mean(directionalfft1,axis=1)
mean2 = np.mean(directionalfft2,axis=0)

vals1=np.linspace(.001,len(mean1),len(mean1))
vals1=len(vals1)/vals1
vals2=np.linspace(.001,len(mean2),len(mean2))
vals2=len(vals1)/vals2

plt.plot(vals1,np.mean((abs(directionalfft1)),axis=1),label='vertical')
plt.plot(vals2,np.mean((abs(directionalfft2)),axis=0),label='horizontal')
plt.legend()
plt.xlim(3,100)
plt.ylim(0,6000)
print(mean1.shape)
print(mean2.shape)

#%%
from scipy import signal
norm = maskedpatch-np.mean(maskedpatch)
testc = autocorrelatefinder(maskedpatch-np.mean(maskedpatch))
plt.imshow(testc)
#%%
corrleft = signal.correlate(norm[:,362],norm[:,362])
corrleft = corrleft/np.max(corrleft)
corrright = signal.correlate(norm[:,1421],norm[:,1421])
corrright = corrright/np.max(corrright)
plt.plot(corrleft,label='left')
plt.plot(corrright,label='right')
plt.legend()
#%%
polar=cv2.linearPolar(maskedpatch,[842,688],900,cv2.WARP_FILL_OUTLIERS)
plt.imshow(polar)
print(maskedpatch.shape)
print(polar.shape)

#%%
from scipy.interpolate import splrep, splev




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


fig, (ax1, ax2) = plt.subplots(1, 2)

test1 = leadtrailfingerfind(polar, 35)

ax1.imshow(test1[0][1])
ax2.imshow(test1[1][1])
#%%
def xgenerator(initial,inputimage):
	'''
	Generates the x arrays for the cross correlation (correcting for r)
	initial is the starting r
	size is the number of theta values
	'''
	
	thetalen = inputimage.shape[0]
	rlen = inputimage.shape[1]
	
	#cross correlation goes from negative to positive
	x0 = np.linspace(-thetalen+1,thetalen-1,num=2*thetalen-1)
	
	#radius needs to be added
	rvals = np.linspace(initial,initial+rlen-1,num = rlen)
	
	xall = np.outer(x0,rvals)

	return xall
xmat = xgenerator(test1[1][0],test1[1][1])
plt.imshow(xmat)

#%%
'''
Completing the cross correlation sample and sample of how to get it back to real lengths
'''
testa = test1[1][1][:,0]
testa = testa-np.mean(testa)

testb = test1[1][1][:,-1]
testb=testb-np.mean(testb)

smallcorr1 = signal.correlate(testa,testa)
smallcorr1 = smallcorr1/np.max(abs(smallcorr1))
smallcorr2 = signal.correlate(testb,testb)
smallcorr2 = smallcorr2/np.max(abs(smallcorr2))

plt.plot(xmat[:,1],smallcorr1,label='close')
plt.plot(xmat[:,-1],smallcorr2,label='far')

#%%
from scipy import interpolate
def averagecorr(inputimage,xmatrix):
	#shift each line by the mean to get rid of DC shift
	linemeans = np.mean(inputimage,axis=0)
	shiftedim = np.subtract(inputimage,linemeans)
	
	allinterps = [None]*inputimage.shape[1] #Create list to store interp functions
	
	for i in range(inputimage.shape[1]):
		x = xmatrix[:,i]
		y = signal.correlate(shiftedim[:,i],shiftedim[:,i])
		y = y/np.max(y)
		allinterps[i] = interpolate.interp1d(x, y, kind='linear', axis=-1, assume_sorted=True)
		
	return allinterps

allinterps = averagecorr(test1[1][1],xmat)
plt.plot(xmat[:,0],allinterps[0](xmat[:,0]))
plt.plot(xmat[:,-1],allinterps[-1](xmat[:,-1]))
#%%
x_all = np.linspace(0, 10, num=101, endpoint=True)
# put all fits to one matrix for fast mean calculation
data_collection = np.vstack((f1_int,f2_int,f3_int))

# calculating mean value
f_avg = np.average(data_collection, axis=0)


#%%
cctest = test1[1][1]


linemeans = np.mean(cctest,axis=0)
test = np.subtract(cctest,linemeans)
corrtest = signal.correlate(test,test)


plt.plot(corrtest[1])
#%%
teststuff = plt.imshow(signal.correlate(cctest,cctest))
plt.imshow(teststuff)
#%%

interptest = interpolate.interp1d(xmat, y, kind='linear', axis=-1, assume_sorted=True)
#%%
smallcorr1 = signal.correlate(test1,test1)
smallcorr1 = smallcorr1/np.max(abs(smallcorr1))
smallcorr2 = signal.correlate(test2,test2)
smallcorr2 = smallcorr2/np.max(abs(smallcorr2))
plt.imshow(smallcorr2)
#%%
n1=smallcorr1.shape[0]
n2=smallcorr2.shape[0]
x1s = np.linspace(-n1/2,n1/2,num=n1)
x2s = np.linspace(-n2/2,n2/2,num=n2)
y1s = np.mean(smallcorr1,axis=1)
y1s = y1s/np.max(y1s)
y2s = np.mean(smallcorr2,axis=1)
y2s = y2s/np.max(y2s)
plt.plot(x1s,y1s,label='right')
plt.plot(x2s,y2s,label='left')
plt.legend()
#%%
extents = tightcropper(
	largestpatchfind(
	thresholdfinder(
		correctedimages[1])),False)

tftbase = fftfinder(correctedimages[0][extents[2]:extents[3],extents[0]:extents[1]])

plt.imshow(np.abs(centerzoomer(tft,[100,100])))
#%%
fig, (ax1, ax2) = plt.subplots(1, 2)

tft1 = fftfinder(mainpatches[0])
tft2 = fftfinder(mainpatches[1])

ax1.imshow(np.abs(centerzoomer(tft1,[100,100])))
ax1.set_title('static')
ax2.imshow(np.abs(centerzoomer(tft2,[100,100])))
ax2.set_title('motion')
#%%
halftest = mainpatches[1]
half=int(len(halftest[0,:])/2)
tftleft = fftfinder(halftest[:,:half],3)
tftright = fftfinder(halftest[:,half:],3)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(centerzoomer(np.abs(tftleft),[100,100]))
ax1.set_title('left')
ax2.imshow(centerzoomer(np.abs(tftright),[100,100]))
ax2.set_title('right')
#To get from pixels from center to wavelength
'''
Divide the length of the image axis by the distance from the DC center value
'''
#%%
plt.imshow(centerzoomer(np.abs(tftleft)-np.abs(tftright),[100,100]))
#%%
prof = radial_profile(abs(tftleft), [int(x/2) for x in tftleft.shape])
prof2 = radial_profile(abs(tftright), [int(x/2) for x in tftleft.shape])
plt.plot(prof,label='left')
plt.plot(prof2,label='right')
plt.legend()
#%%
def polarimage(inputimage):
	img = abs(inputimage)
	ro,col=img.shape
	cent=(int(col/2),int(ro/2))
	max_radius = int(np.sqrt(ro**2+col**2)/2)
	polar=cv2.linearPolar(img,cent,max_radius/5,cv2.WARP_FILL_OUTLIERS)
	polar = (polar[:cent[1]]+polar[cent[1]:])/2
	return polar


right = polarimage(tftright)
left = polarimage(tftleft)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.imshow(left)
ax1.set_title('left')
ax2.imshow(right)
ax2.set_title('right')

#%%
vals=np.linspace(.001,len(right[0,:]),len(right[0,:]))
plt.plot(vals,np.mean(right[340:400],axis=0),label='right')
plt.plot(vals,np.mean(left[340:400],axis=0),label='left')

plt.legend()


#%%

vals=np.linspace(.001,len(right[0,:]),len(right[0,:]))
plt.plot(len(right[0,:])/vals,np.mean(right[300:450],axis=0),label='right')
plt.plot(len(right[0,:])/vals,np.mean(left[300:450],axis=0),label='left')
plt.xlim(0,100)
plt.legend()

#%%


#%%

#%%
leveled = movingim-fitfunc(r1,*popt)
leveled=leveled-np.min(leveled)
leveled=leveled.astype(np.uint8)

img=leveled
img= cv2.GaussianBlur(img,(5,5),0)
ret,th1 = cv2.threshold(img,25,255,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
plt.imshow(movingim,cmap='gray')
plt.imshow(th1,alpha=0.3)  
#cv2.drawContours(th1, contours, -1, (0,255,0), 3)


#%%
plt.imshow(movingim)