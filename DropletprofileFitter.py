'''
Functions to find properties of droplet such as contact angle etc
'''
def convtolocs()
	'''
	simply renaming np.argwhere
	Takes the image file of 1s and 0s from edge detection and
	turns it into coordinates
	'''
 	return np.argwhere(edgedetect)

def circle(x,a,b,r):
	'''
	Simply a function for a circle
	'''
    return np.sqrt(r**2-(x-a)**2)+b


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

def edgeinfofinder(locs,left,pixelbuff,circfitguess,zweight):
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
    popt, pcov = curve_fit(circle, trimDat[:,1], trimDat[:,0],p0=circfitguess, sigma=sigma,maxfev=5000)
    def paramcirc(x):
        return circle(x,*popt)
    mcirc=derivative(paramcirc,0)
    #Return angle in degrees
    thet=np.arctan(mcirc)*180/np.pi
    #Shift circle back
    popt=popt+[contactx,contacty,0]
    return [contactx,contacty,popt,thet,mcirc]