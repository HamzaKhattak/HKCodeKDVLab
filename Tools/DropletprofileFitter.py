'''
Functions to find properties of droplet such as contact angle etc
'''
import numpy as np
from scipy.optimize import curve_fit
from scipy.misc import derivative
from Tools.EdgeDetection import *

def circle(x,a,b,r):
	'''
	Simply a function for a circle
	'''
	return np.sqrt(r**2-(x-a)**2)+b

def pol2ndorder(x,a,b,c):
	return a+b*x+c*x**2

def pol3rdorder(x,a,b,c,d):
	return a+b*x+c*x**2+d*x**3

def pol4thorder(x,a,b,c,d,e):
	return a+b*x+c*x**2+d*x**3+e*x**4

def slopeptline(x,m,x0,y0):
	'''
	slope point form for a line
	'''
	return m*(x-x0)+y0

def linef(x,m,b):
	'''
	equation of a line for fitting
	'''
	return m*x + b


def splitlinefinder(locs,centerybuff):
	'''
	Finds the rough center location based on some buffer
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
	    contactloc=np.argmin(locs[:,0])
	else:
	    contactloc=np.argmax(locs[:,0])

	contactx=locs[contactloc,0]
	contacty=locs[contactloc,1]

	#Set up trimmed Data set for fit using buffered area and only positive values
	#Will need to change to also include data from reflection
	if left==True:
	    conds=np.logical_and.reduce((locs[:,0]<contactx+pixelbuff[0],locs[:,1]>contacty,locs[:,1]<pixelbuff[1]))
	else:
	    conds=np.logical_and.reduce((locs[:,0]>contactx-pixelbuff[0],locs[:,1]>contacty,locs[:,1]<pixelbuff[1]))
	    
	trimDat=locs[conds]-[contactx,contacty]

	#Set up weighting
	sigma = np.ones(len(trimDat[:,1]))
	sigma[np.argmin(trimDat[:,1])] = zweight
	#The fitter is annoyingly dependant on being close to the actual parameters values to get a good guess
	popt, pcov = curve_fit(fitfunction, trimDat[:,0], trimDat[:,1],p0=fitguess, sigma=sigma,maxfev=5000)
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
	shiftedarray=torotate-[ox,oy]
	rotatedarray=np.zeros(shiftedarray.shape)
	#Apply rotation transform
	rotatedarray[:,0] = np.cos(angle) * shiftedarray[:,0] - np.sin(angle) * shiftedarray[:,1]
	rotatedarray[:,1] = np.sin(angle) * shiftedarray[:,0] + np.cos(angle) * shiftedarray[:,1]
	return rotatedarray+[ox,oy]

def angledet(x1,y1,x2,y2):
	'''
	determine angle needed to rotate to get line horizontal from an x and y point
	'''
	dx=x2-x1
	dy=y2-y1
	return np.arctan(dy/dx)

def linedet(MultipleEdges):
	'''
	Takes a python array of edge xy locations and returns a list of endpoints for use with fitting
	'''
	#Create empty arrays
	numIm=len(MultipleEdges)
	leftxy=np.zeros([numIm,2])
	rightxy=np.zeros([numIm,2])
	for i in range(numIm):
		#Get the indexes of the minimum and maximum x points
		#Can be modified to extract some other property from each of the xy arrays
		leftindex = np.argmin(MultipleEdges[i][:,0])
		rightindex = np.argmax(MultipleEdges[i][:,0])
		leftxy[i] = [MultipleEdges[i][leftindex,0], MultipleEdges[i][leftindex,1]]
		rightxy[i] = [MultipleEdges[i][rightindex,0], MultipleEdges[i][rightindex,1]]
	return leftxy, rightxy



