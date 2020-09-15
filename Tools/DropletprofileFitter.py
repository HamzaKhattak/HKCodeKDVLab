'''
Functions to find properties of droplet such as contact angle etc
'''
import numpy as np
from scipy.optimize import curve_fit
from scipy.misc import derivative
from Tools.EdgeDetection import *
from skimage import measure

'''
This first part is for the functions relevant to the side profile
'''
def circle(x,a,b,r):
	'''
	Simply a function for a circle
	'''
	return np.sqrt(r**2-(x-a)**2)+b

def pol2ndorder(x,a,b,c):
	return a+b*x+c*x**2

def pol2nolin(x,a,b):
	return a + b*x**2

def pol3rdorder(x,a,b,c,d):
	return a+b*x+c*x**2+d*x**3

def pol4thorder(x,a,b,c,d,e):
	return a+b*x+c*x**2+d*x**3+e*x**4

def slopeptline(x,m,x0,y0):
	'''
	slope point form for a line
	'''
	return m*(x-x0)+y0

def linfx(x,m,b):
	'''
	equation of a line
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

def contactptfind(locs,left,buff=0,doublesided=False):
	'''
	This function finds the contact pt between the droplet and surface
	It takes an array of xy pairs and a left Boolean True False to determine side
	It also takes a buffer argument to not include points within a buffer from the top
	It returns the contactx and contacty positions
	'''
	miny = np.amin(locs[:,1])
	maxy = np.amax(locs[:,1])
	appproxsplity=np.mean(locs[:,0])

	if left==True:
		trimDat = locs[locs[:,0] < appproxsplity]
	else:
	    trimDat = locs[locs[:,0] > appproxsplity]
			
	#Account for double sided buffer requirements (ie if it is mirrored)
	if doublesided == True:
		conds2 = np.logical_and(trimDat[:,1] > miny + buff, trimDat[:,1] < maxy - buff)
		trimDat = trimDat[conds2]
		#Fit a parabola to the data with the x and y flipped
		popt, pcov = curve_fit(pol2ndorder, trimDat[:,1], trimDat[:,0])
		#Positive curvature means the contact point is a minimum and negative means it is a max
		if (popt[-1] > 0):
			contactx = np.amin(trimDat[:,0])
		else:
			contactx = np.amax(trimDat[:,0])

	else:
		conds2 = trimDat[:,1] < maxy - buff
		trimDat = trimDat[conds2]
		contactx = trimDat[np.argmin(trimDat[:,1]),0]
	#Find y values (need to account for multiple mins)
	allcens = np.argwhere(locs[:,0] == contactx)
	contacty = np.mean(locs[allcens,1])
	return contactx, contacty


def datafitter(locs,left,pixelbuff,zweight,fitfunction,fitguess,axisflip=False):
	'''
	This function takes a numpy array of edge location xy values and returns
	the location of the contact point as well as a fitted function
	Parameters:
	    locs: the input array
	    left: True for left side of droplet, false for right
	    pixelbuff: How many pixels in xy to include for fit plus a y buffer
	    fitguess: Guess's for fit parameters
		fitfunction: the function used for fitting
	    zweight: anything below 1 gives extra weight to the zero
		axisflip: in some cases it makes sense to flip x and y for better fitting
		If flipped, the popt will be for y as a function of x
		

	Circle fitting can be a bit buggy, need to be fairly close with parameters.
	Better to overestimate radius somewhat.
	'''
	appproxsplity=np.mean(locs[:,0])
	contactx, contacty = contactptfind(locs,left,buff=pixelbuff[2])


	#Set up trimmed Data set for fit using buffered area and only positive values
	#Will need to change to also include data from reflection
	if left==True:
		conds1 = locs[:,0] < appproxsplity
	else:
	    conds1 = locs[:,0] > appproxsplity
	
	conds2 = np.logical_and.reduce((np.abs(locs[:,0]-contactx)<pixelbuff[0],locs[:,1]>contacty,locs[:,1]<contacty+pixelbuff[1]))

	conds=np.logical_and(conds1,conds2)
	trimDat=locs[conds]-[contactx,contacty]
	#Set up weighting
	sigma = np.ones(len(trimDat[:,1]))
	sigma[np.argmin(trimDat[:,1])] = zweight
	
	#Flip to avoid issues with vertical regions if needed
	if axisflip==True:
		trimDat = np.flip(trimDat,axis=1)
	#The fitter is annoyingly dependant on being close to the actual parameters values to get a good guess
	popt, pcov = curve_fit(fitfunction, trimDat[:,0], trimDat[:,1],p0=fitguess, sigma=sigma,maxfev=5000)

	def paramfunc(x):
	    return fitfunction(x,*popt)
	m0=derivative(paramfunc,0)
	#Return angle in degrees
	if axisflip==False:
		thet=np.arctan(m0)*180/np.pi
	else:
		thet=np.arctan(1/m0)*180/np.pi
	return [contactx,contacty,thet,m0,popt,pcov]

def flipper(toflip,x1,y1,x2,y2):
	'''
	Flips an array of xy points about a line defined by x1,x2,y1,y2
	'''
	#Create line to flip
	l=np.array([x2-x1,y2-y1])
	#Shift array to be flipped so they are vectors from origin of l
	shiftarray= toflip - [x1,x2]
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

def xflipandcombine(toflip,flipy):
	'''
	Flips an already rotated edge point array and combines the top and the bottom
	#flipy is where to flip 
	'''

	#Seperate and flip the negative values
	topvalues = toflip[toflip[:,1]>flipy]
	bottomvalues = toflip[toflip[:,1]<flipy]*[1,-1]+[0,2*flipy]
	return np.concatenate([topvalues,bottomvalues])


def angledet(x1,y1,x2,y2):
	'''
	determine angle needed to rotate to get line horizontal from an x and y point
	'''
	dx=x2-x1
	dy=y2-y1
	return np.arctan(dy/dx)

def linedet(MultipleEdges, buff = 0):
	'''
	Takes a python array of edge xy locations and returns a list of endpoints for use with fitting
	ylims is if there is a need to only search in a certain location
	'''
	#Create empty arrays
	numIm=len(MultipleEdges)
	leftxy=np.zeros([numIm,2])
	rightxy=np.zeros([numIm,2])
	#Subtract background if needed and select image, droplet should be high so invert
	
	for i in range(numIm):
		#Get the indexes of the minimum and maximum x points
		#Can be modified to extract some other property from each of the xy arrays (ie other than argmin)
		contactxl, contactyl = contactptfind(MultipleEdges[i],True,buff,True) #left edge
		contactxr, contactyr = contactptfind(MultipleEdges[i],False,buff,True) #right edge
		#Get into the right form
		leftxy[i] = [contactxl, contactyl]
		rightxy[i] = [contactxr,contactyr]
	
	'''
	OLD CODE, not used anymore but has useful snippet for getting arguments in masked array
	
	#Define center and buffer range
	ycen=(ylims[0]+ylims[1])/2
	yran=np.abs(ylims[1]-ylims[0])/2
	for i in range(numIm):
		#Select a slice in y to analyze
		cond = np.abs(MultipleEdges[i][:,1]-ycen) < yran #the condition to limit to the yrange of interest
		lefttrimidx = np.where(cond)[0] #Get the indices with the condition
		leftindex = lefttrimidx[MultipleEdges[i][:,0][lefttrimidx].argmin()] #Find the argument of the minimum in that region
		rightindex = lefttrimidx[MultipleEdges[i][:,0][lefttrimidx].argmax()] #Same for the right
		#Get the actual values
		leftxy[i] = [MultipleEdges[i][leftindex,0], MultipleEdges[i][leftindex,1]]
		rightxy[i] = [MultipleEdges[i][rightindex,0], MultipleEdges[i][rightindex,1]]
	'''
	return leftxy, rightxy

def thetdet(edgestack,buff=0):
	'''Find the angle to rotate a stack of images based on the location of droplet edges
	Returns the angle and a point with which to rotate'''
	leftlineinfo, rightlineinfo = linedet(edgestack,buff)
	#Contact points needed for rotation, actually return the ones from the datafitter
	allcontactpts=np.concatenate([leftlineinfo,rightlineinfo])

	#Fit a line to the contact points 
	fitlineparam,firlinecov = curve_fit(linfx,allcontactpts[:,0],allcontactpts[:,1])

	#create 2 random points based on the line for the angle detection function
	leftedge=[leftlineinfo[0,0],linfx(leftlineinfo[0,0],*fitlineparam)]
	rightedge=[rightlineinfo[0,0],linfx(rightlineinfo[0,0],*fitlineparam)]
	#Find the angle with horizontal
	thet=angledet(*leftedge,*rightedge)
	return thet, leftedge

def edgestoproperties(edgestack,lims,fitfunc,fitguess,axisflip=True):
	'''
    Takes a edgestack and returns a list of angles for the right and left 
    positions and angles
    edgestack is a python list of numpy arrays containing the edges
    lims is the x and y pixel range to use for fitting plus a third value to trim the endpt search
    fitfunc is the function to use to find the angle
	fitguesss is the guess for those parameters
	ylims is optional for when need to select a specific y region because the pipette is farther than the droplet
    '''
	#Create arrays to store data
	numEd=len(edgestack)
	dropangle = np.zeros([numEd,2])
	contactpts = np.zeros([numEd,2,2])
	paramlist=np.zeros([numEd,2,len(fitguess)])

	thetatorotate, leftedge = thetdet(edgestack,buff = lims[2])
	rotateprop = [thetatorotate,leftedge]

	for i in range(numEd):
		rotatededges=rotator(edgestack[i],-thetatorotate,*leftedge)
		combovals=xflipandcombine(rotatededges,leftedge[1])
		fitl=datafitter(combovals,True,lims,1,fitfunc,fitguess,axisflip=True)
		fitr=datafitter(combovals,False,lims,1,fitfunc,fitguess,axisflip=True)
		dropangle[i] = [fitl[2],fitr[2]]
		contactpts[i] = [[fitl[0],fitl[1]],[fitr[0],fitr[1]]]
		paramlist[i] = [fitl[-2],fitr[-2]]
	return dropangle, contactpts, paramlist, rotateprop


'''
This section is for function relevant to the top profile
'''
def eggexpr(X,Y):
	'''
	This expression returns the left side of the equation of an egg
	The right side should equal 1

	'''
	return np.array([X**2, X * Y, Y**2, X, Y,X*Y**2,Y*X**2])

def eggfitter(x,y,eggdef = eggexpr):
	'''
	returns paramameters of an agg git for a given datset
	the eggexprtion should = 1
	'''
	X, Y = x[:,np.newaxis], y[:,np.newaxis]
	# Formulate and solve the least squares problem ||Ax - b ||^2
	A = np.hstack(eggexpr(X,Y))
	b = np.ones_like(X)
	x1, resid, rnk, sing = np.linalg.lstsq(A, b,rcond=None)
	param=x1.squeeze()
	paramresid= resid.squeeze()
	return param, paramresid


def contourfinder(x,y,param,eggdef = eggexpr):
	'''
	Given x and y list (only used for length) as well as parameters for a fit this function gives
	an x list and y list of the countours
	Again, right side should be 1
	'''
	x_coord = np.arange(len(x))
	y_coord = np.arange(len(y))
	X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
	
	temparray = eggdef(X_coord,Y_coord) #Calculate the x,y dependant bits of the sum
	temparray=param[:,None,None]*temparray #Multiply in the parameters
	Z_coord = np.sum(temparray,axis=0) #Add each of the terms together
	
	contours = measure.find_contours(Z_coord, 1) #Find when 1
	cx = contours[0][:,1]
	cy = contours[0][:,0]
	return cx, cy
	

def arc_length(x,y):
	'''
	Calculates arclength for a given dataset using trapezoidal summing
	Data must be properly ordered
	'''
	arc = np.sqrt(np.gradient(x)**2+np.gradient(y)**2)
	arc = np.trapz(arc)
	return arc

def areafind(x,y):
	'''
	Use Green's Theorum to find from edges integrate along curve: 0.5(x dy - y dx)
	'''
	return 0.5*np.trapz(x*np.gradient(y)-y*np.gradient(x))

def PolyArea(x,y):
	'''
	This method used the 'shoelace method'
	'''
	return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def comboperimcalc(XYdat,eggdef=eggexpr):
	'''
	This function outputs the parameters,residuals, fit data and arclength
	for a shape given the edge data in  [x1,y1],[x2,y2]... form
	'''
	x, y = XYdat[:,0] , XYdat[:,1]
	meanx = np.mean(x)
	meany = np.mean(y)
	param, resid = eggfitter(x, y,eggdef)
	cx, cy = contourfinder(x,y,param)
	arc = arc_length(cx,cy)
	area = areafind(cx,cy)
	return param, resid, [cx,cy], [meanx,meany], arc, area
	
def seriescomboperimcalc(XYTimeSeries,eggdef=eggexpr):
	'''
	Repeats the perimeter calculation for a series of images
	'''
	mean = np.zeros([len(XYTimeSeries),2])
	arc = np.zeros(len(XYTimeSeries))
	area = np.zeros(len(XYTimeSeries))
	for i in range(len(XYTimeSeries)):
		dat = comboperimcalc(XYTimeSeries[i],eggdef)
		mean[i], arc[i], area[i] = dat[3:]
	return mean, arc, area