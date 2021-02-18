import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import medfilt

'''
def smoothingfilter(data,windowparam=4,polyorder=3,fraction=False):
	
	# Simply smooths data based on a window fraction (ie what 
	# fraction of total data should the window be) or by simply rounding to odd
	if fraction==True:☻
		arlen=data.size #Get size
		windowlength=arlen/windowparam #divid
		windowlength=np.ceil(windowlength) // 2 * 2 + 1 #make odd
		windowlength=int(windowlength) #Make into an int
	else:
		windowlength=np.ceil(windowparam) // 2 * 2 + 1 #make odd
		windowlength=int(windowlength) #Make into an int
	return savgol_filter(data,windowlength,polyorder) #filter

'''
def smoothingfilter(data,windowparam=4):
	'''
	Simply smooths data based on a window fraction (ie what
	fraction of total data should the window be) or by simply rounding to odd
	'''

	return medfilt(data,windowparam) #filter

def anglefilter(data,windowsize=21,polyorder=3):
	'''
	Windowsize must be odd
	'''
	result = savgol_filter(np.abs(data),windowsize,polyorder)
	return result

def plateaufilter(timearray,forcearray,regionofinterest,smoothparams=[],sdevlims=[0.1,0.2],outlierparam=1):
	'''
	This function finds the high or low plateaus in the force curves
	It takes a time array, force array, which distance to cut the tail, smoothing parameters
	and limits for what fraction of standard deviation to use in the velocity 
	and acceleration cutoffs
	returns arrays with the smoothed data, vels, accs, topfiltered and bottomfiltered
	arrays, the final filtered arrays are in [t1,y1],[t2,y2] format
	'''
	
	#Smooth and get velocities accelerations
	dt = timearray[1] - timearray[0]
	cutindexl = (np.abs(timearray - regionofinterest[0])).argmin()
	cutindexr = (np.abs(timearray - regionofinterest[1])).argmin()
	cutTime = timearray[cutindexl:cutindexr]
	if smoothparams != [0,0]:
		smootheddat=smoothingfilter(forcearray[cutindexl:cutindexr],*smoothparams)
		uncutsmooths=smoothingfilter(forcearray,*smoothparams)
	else:
		smootheddat=forcearray[cutindexl:cutindexr]
		uncutsmooths=forcearray
	vels=np.gradient(smootheddat,dt)
	accs=np.gradient(vels,dt)
	
	#Set velocity and acceleration limits and filter data based on those
	velLim=sdevlims[0]*np.std(vels)
	accLim=sdevlims[1]*np.std(accs)
	filtcond= (np.abs(vels)<velLim) & (np.abs(accs)<accLim) 
	filtered2=smootheddat[filtcond]
	modtimes=timearray[cutindexl:cutindexr]
	filteredtimes2=modtimes[filtcond]
	
	#Find the high and plateaus
	#High (assumes approximate symettry around 0)
	filterhigh = filtered2 > 0
	
	meanhigh1 = np.mean(filtered2[filterhigh])
	meanhsdev = np.std(filtered2[filterhigh])

	highcond = np.abs(filtered2 - meanhigh1) < outlierparam*meanhsdev
	high = np.transpose([filteredtimes2[highcond],filtered2[highcond]])
	
	
	#Repeat for low
	filterlow=filtered2<0

	meanlow1 = np.mean(filtered2[filterlow])
	meanlsdev = np.std(filtered2[filterlow])

	lowcond = np.abs(filtered2 - meanlow1) < outlierparam*meanlsdev
	low = np.transpose([filteredtimes2[lowcond],filtered2[lowcond]])

	
	#Find the indexes in timearray which are part of low and high
	idl = np.isin(timearray, filteredtimes2[lowcond], assume_unique=True)
	idh = np.isin(timearray, filteredtimes2[highcond], assume_unique=True)
	
	#Return numpy list with data
	return [cutTime,uncutsmooths,vels,accs,[high,low],[idh,idl]]



def clusteranalysis(data,separam):
	#find mean and standard deviation for whole result
	#Input should be in [[x1,y1],[x2,y2]...] form
	meanwhole=np.mean(data[:,1])
	sdevwhole=np.std(data[:,1])

	#Find mean and standard deviation using each plateau
	diffs = np.gradient(data[:,0])
	meandiff = np.mean(diffs)
	sdevdiffs = np.std(diffs)
	yvals = data[:,1]
	#Find for each individual jump
	jumplocs = np.argwhere(np.abs(diffs-meandiff)>separam*sdevdiffs)
	jumplocs = np.insert(jumplocs, 0, 1)
	jumplocs = np.insert(jumplocs, len(jumplocs) ,len(yvals) - 1)
	numjumps=len(jumplocs)
	statsclust=np.zeros([numjumps,2])
	for i in range(numjumps-1):
		statsclust[i,0]=np.mean(yvals[jumplocs[i]+1:jumplocs[i+1]-1])
		statsclust[i,1]=np.std(yvals[jumplocs[i]+1:jumplocs[i+1]-1])
	clusterm=np.mean(statsclust[:,0])
	clustersdev=np.std(statsclust[:,0])
	clusterserr=clustersdev/len(statsclust[:,0])
	return [meanwhole,sdevwhole],[clusterm,clustersdev,clusterserr],statsclust,diffs,jumplocs


def rejectoutliers(dat,m=2):
	'''
	Returns indices that are not from a numpy list
	'''
	return abs(dat-np.mean(dat)) < m*np.std(dat)
