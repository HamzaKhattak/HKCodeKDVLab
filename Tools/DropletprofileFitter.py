'''
Functions to find properties of droplet such as contact angle etc
'''
import numpy as np
from scipy.optimize import curve_fit
from scipy.misc import derivative

def circle(x,a,b,r):
	'''
	Simply a function for a circle
	'''
	return np.sqrt(r**2-(x-a)**2)+b

def pol3rdorder(x,a,b,c,d):
	return a+b*x+c*x**2+d*x**3


def slopeptline(x,m,x0,y0):
	'''
	slope point form for a line
	'''
	return m*(x-x0)+y0


def splitlinefinder(locs,centerybuff):
	'''
	Finds the center location based on some buffer
	'''
	#Which area to use for splitting line (ie how far up y)
	splitlineavregion=np.max(locs[:,0])-centerybuff
	splitline=np.mean(locs[:,1][locs[:,0]>splitlineavregion])
	return splitline

def datafitter(locs,left,pixelbuff,zweight,fitfunction,fitguess):
	'''
	This function takes a numpy array of edge location xy values and returns
	the location of the contact point as well as a fitted circle
	Parameters:
	    locs: the input array
	    left: True for left side of droplet, false for right
	    pixelbuff: How many pixels in x to include for fit
	    circfitguess: Guess's for circle fit parameters, make sure to make negative for
	        right side. [xcenter,ycenter,radius]
	    zweight: anything below 1 gives extra weight to the zero

	Circle fitting can be a bit buggy, need to be fairly close with parameters.
	Better to overestimate radius somewhat.
	'''

	#Get the min or max position
	if left==True:
	    contactloc=np.argmin(locs[:,1])
	else:
	    contactloc=np.argmax(locs[:,1])

	contactx=locs[contactloc,1]
	contacty=locs[contactloc,0]

	#Set up trimmed Data set for fit using buffered area and only positive values
	#Will need to change to also include data from reflection
	if left==True:
	    conds=np.logical_and(locs[:,1]<contactx+pixelbuff,locs[:,0]>contacty)
	else:
	    conds=np.logical_and(locs[:,1]>contactx-pixelbuff,locs[:,0]>contacty)
	    
	trimDat=locs[conds]-[contacty,contactx]

	#Set up weighting
	sigma = np.ones(len(trimDat[:,0]))
	sigma[np.argmin(trimDat[:,0])] = zweight
	#The fitter is annoyingly dependant on being close to the actual parameters values to get a good guess
	popt, pcov = curve_fit(fitfunction, trimDat[:,1], trimDat[:,0],p0=fitguess, sigma=sigma,maxfev=5000)
	def paramfunc(x):
	    return fitfunction(x,*popt)
	m0=derivative(paramfunc,0)
	#Return angle in degrees
	thet=np.arctan(m0)*180/np.pi
	return [contactx,contacty,popt,thet,m0]

def flipper(toflip,x1,y1,x2,y2):
	'''
	Flips an array of xy points about a line defined by x1,x2,y1,y2
	'''
	#Create line to flip
	l=np.array([x2-x1,y2-y1])
	#Shift array to be flipped so they are vectors from origin of l
	shiftarray=toflip-[x1,x2]
	projection = l * np.dot(toflip, l) / np.dot(l, l)
	return 2*projection-shiftarray

def rotator(torotate,angle,ox,oy):
	"""
	Rotate an x,y list of points counterclockwise by a given angle around a given origin.
	"""
	'''
	ox, oy = origin
	px, py = point

	qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
	qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
	'''
	#shift array to origin
	rotatedarray=torotate-[ox,oy]
	#Apply rotation transform
	rotatedarray[:,0] = ox + np.cos(angle) * shiftarray[:,0] - np.sin(angle) * shiftarray[:,1]
	rotatedarray[:,1] = ox + np.sin(angle) * shiftarray[:,0] - np.cos(angle) * shiftarray[:,1]
	return rotatedarray

def angledet(x1,y1,x2,y2):
	'''
	determine angle needed to rotate to get line horizontal
	'''
	dx=x2-x2
	dy=y2-y1
	return np.arctan(dy,dx)