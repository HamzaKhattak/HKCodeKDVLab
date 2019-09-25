import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
'''
takes in two arrays and finds the cross correlation array of shifts
'''

def crosscorrelator(a,b):
	'''
	This function takes in two 1D arrays a and b, normalizes them
	to find the cross correlation of a with b and then returns and
	returns an [x,y] list with the index of 
	'''
	#Normalize the input vectors
	norma = (a - np.mean(a)) / (np.std(a) * len(a))
	normb = (b - np.mean(b)) / (np.std(b))
	#Use numpy correlate to find the correlation
	corry = np.correlate(norma, normb, 'full')
	#Shift back to find the x array
	corrx = np.arange(2*len(norma)-1)-(len(norma)-1)
	return np.transpose([corrx,corry])

def gaussfunc(x,a,mu,sig):
	return a*np.exp((-(x-mu)**2)/(2*sig))


def centerfinder(vecx,vecy,buff):
	'''
	This function takes a 1D vector and fits a gaussian to its max
	peak. The buff (an integer) argument decides how many points to use around the
	max value
	'''
	#Find where the max peak is generally
	maxpos=np.argmax(vecy)
	#Find the 2 edges to use for this fit and trim the data
	lefte=maxpos-buff
	righte=maxpos+buff
	xdata=vecx[lefte:righte]
	ydata=vecy[lefte:righte]
	#Perform the curve fit, guess parameters, just since maxpos
	# will be pretty close to the center
	popt, pcov = curve_fit(gaussfunc,xdata,ydata,p0=[1,vecx[maxpos],2*buff])
	#Find standard deviation in parameters
	perr = np.sqrt(np.diag(pcov))
	#Return parameter and standard deviation
	return popt, perr

def xvtfinder(images,baseimage,cutloc,gausspts1):
	'''
	Takes a image sequence and the original image and returns series of shifts
	as well as the full cross correlation arrays
	from the base image using cross correlation at the y pixel defined by cutloc
	gaussspts1 is the number of points to use in the gaussian fit on either side
	'''
	imdim=images.ndim
	#Account for single image case
	if imdim==2:
		images=np.expand_dims(images, 0)

	#Create empty arrays to store data
	centerloc=np.zeros([images.shape[0],2])
	alldat=np.zeros([images.shape[0],images.shape[2]*2-1,2])
	#autocorrelation for base
	basecut=baseimage[cutloc]
	basecorr=crosscorrelator(basecut,basecut)
	bgparam, bgerr = centerfinder(basecorr[:,0],basecorr[:,1],gausspts1)
	#Perform cross correlation and use gaussian fit to find center position
	for i in range(images.shape[0]):
		alldat[i] = crosscorrelator(images[i,cutloc],basecut)
		gparam, gerr = centerfinder(alldat[i,:,0],alldat[i,:,1],gausspts1)
		centerloc[i]=[gparam[1],gerr[1]]
	#Account for the 0 point
	centerloc = centerloc-[bgparam[1],0]
	return centerloc, alldat

