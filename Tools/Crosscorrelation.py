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

def ccorrf(x, y, unbiased=True, demean=True, sym = True):
   ''' JC's crosscoorrelation function for 1D

   Parameters
   ----------
   x, y : arrays
      time series data
   unbiased : boolean
      if True, then denominators is n-k, otherwise n
   sym : boolean
       if True, the outpur is symmetrical (lag in both positive and negative)
   Returns
   -------
   ccorrf : array
      cross-correlation array

   Notes
   -----
   This uses np.correlate which does full convolution. For very long time
   series it is recommended to use fft convolution instead.
   '''
   n = len(x)
   if demean:
       xo = x - x.mean()
       yo = y - y.mean()
   else:
       xo = x
       yo = y
   if unbiased:
       xi = np.ones(n)
       d = np.correlate(xi, xi, 'full')
   else:
       d = n

   if sym:
       corry = (np.correlate(xo, yo, 'full') / (d*(np.std(x) * np.std(y))))
   else:
       corry = (np.correlate(xo, yo, 'full') / (d*(np.std(x) * np.std(y))))[n - 1:]
   corrx = np.arange(2*len(x)-1)-(len(x)-1)
   return  np.transpose([corrx,corry])

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
	#Find standard error in parameters
	perr = np.sqrt(np.diag(pcov))/np.sqrt(len(xdata))
	#Return parameter and standard deviation
	return popt, perr

def xvtfinder(images,baseimage,cutloc,gausspts1):
    '''
    Takes a image sequence and the original image and returns series of shifts
    from the base image using cross correlation at the y pixel defined by cutloc
    gaussspts1 is the number of points to use in the gaussian fit on either side
    '''
    #Create empty array to store data
    centerloc=np.zeros([images.shape[0],2])
    #Perform cross correlation and use gaussian fit to find center position
    for i in range(croppedimages.shape[0]):
        alldat[i]=crco.crosscorrelator(images[i,cutloc],baseimage)
        gparam, gcov = crco.centerfinder(alldat[i,:,0],alldat[i,:,1],gausspts1)
        gerr = np.sqrt(np.diag(gcov))
        centerloc[i]=[gparam[1],gerr[1]]
    #Account for the 0 point
    centerloc = centerloc-[centerloc[0,0],0]
    return centerloc

