'''
This code performs the edge location and cross correlation analysis across multiple images
'''

import sys, os, glob, pickle, re
import matplotlib.pyplot as plt
import numpy as np
import importlib
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter
#%%
#import similaritymeasures

#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"E:\SoftnessTest\SISThickness2"


os.chdir(CodeDR) #Set  current working direcotry to the code directory


sys.path.append('./Tools') #Add the tools to the system path so modules can be imported

#Import required modules
import DropletprofileFitter as df
importlib.reload(df)
import Crosscorrelation as crco
importlib.reload(crco)
import ImportTools as ito 
importlib.reload(ito)
import EdgeDetection as ede
importlib.reload(ede)
import PlateauAnalysis as planl
importlib.reload(planl)

#Remove to avoid cluttering path
sys.path.remove('./Tools') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)


#%%
#Check the extrema images and note the limits that make sense
noforce=ito.imread2(dataDR+'\\base.tif')
ex1=ito.imread2(dataDR+'\\extreme1.tif')
ex2=ito.imread2(dataDR+'\\extreme2.tif')

gs = gridspec.GridSpec(1, 3)

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax1.imshow(noforce)
ax2.imshow(ex1)
ax3.imshow(ex2)
#%%

#Specify parameters
#Cropping
#Select the minimum (1s) and maximum (2s) crop locations
#Needs to include the pipette ends
x1c=270
x2c=1300
y1c=310
y2c=940
croppoints=[x1c,x2c,y1c,y2c]

#Select crop region for fitting (just needs to be large enough so droplet end is the max)
yanlow=550
yanhigh=700
yanalysisc=[yanlow-y1c,yanhigh-y1c]

croppedbase=ito.cropper(noforce,*croppoints)
croppedex1=ito.cropper(ex1,*croppoints)
croppedex2=ito.cropper(ex2,*croppoints)

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(gs[0, 0])

ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax2.axhline(yanalysisc[0])
ax2.axhline(yanalysisc[1])
ax1.imshow(croppedbase)
ax2.imshow(croppedex1)
ax3.imshow(croppedex2)
#%%

#Cross correlation
cutpoint=30 # y pixel to use for cross correlation
guassfitl=20 # Number of data points to each side to use for guass fit

#Edge detection
imaparam=[-40,20,.05] #[threshval,obsSize,cannysigma]
fitfunc=df.pol3rdorder #function ie def(x,a,b) to fit to find properties
fitguess=[0,1,1,1]
pixrange=[60,60,25] #first two are xy bounding box for fit, last is where to search for droplet tip
#Specify an image to use as a background (needs same dim as images being analysed)
#Or can set to False
background=False 

threshtest=ede.edgedetector(croppedex1,background,*imaparam)
plt.figure()
plt.imshow(croppedex1,cmap=plt.cm.gray)
plt.plot(threshtest[:,0],threshtest[:,1],'r.',markersize=1)
plt.axhline(cutpoint,ls='--')

#%%
'''
Run on all of the images
'''
def split_on_letter(s):
	r'''This code splits at the first letter to allow for sorting based on
	the first letter backslash W is for the nonaplhanumeric and backslash d for
	decimals. The hat then inverts that'''
	match = re.compile("[^\W\d]").search(s)
	return [s[:match.start()], s[match.start():]]

def namevelfind(s):
	'''
	Extracts the speed from the file name based on how it is set up
	Super sketchy but works for now
	'''
	allstrings = split_on_letter(s)
	rawnum=allstrings[0]
	if rawnum[0] == '0':
		result = float(allstrings[0])/10
	else:
		result = float(allstrings[0])
	return result


#Import images
#Use glob to get foldernames, tif sequences should be inside
def foldergen():
	folderpaths=glob.glob(os.getcwd()+'/*/')
	foldernames=next(os.walk('.'))[1]
	#filenames=glob.glob("*.tif") #If using single files
	
	#Empty array for the position vs velocity information
	dropProp=[None]*len(folderpaths)
	#Sort the folders by the leading numbers
	velocitylist1=[namevelfind(i) for i in foldernames]

	foldernames = [x for _,x in sorted(zip(velocitylist1,foldernames))]
	folderpaths = [x for _,x in sorted(zip(velocitylist1,folderpaths))]
	return folderpaths,foldernames,dropProp

folderpaths, foldernames, dropProp = foldergen()
#%%
#Edge detection and save
for i in range(len(folderpaths)):
	imagestack=ito.omestackimport(folderpaths[i])
	croppedimages=ito.cropper(imagestack,*croppoints)
	noshift=croppedbase
	#Find the cross correlation xvt and save to position arrays
	xvals , allcorr = crco.xvtfinder(croppedimages,noshift,cutpoint,guassfitl)
	ito.savelistnp(folderpaths[i]+'correlationdata.npy',[xvals,allcorr])
	
	#Define no shift cropped image as first frame, could change easily if needed
	#Perform edge detection to get python array
	stackedges = ede.seriesedgedetect(croppedimages,background,*imaparam)
	ito.savelistnp(folderpaths[i]+'edgedata.npy',stackedges) #Save for later use
	print(folderpaths[i]+ ' completed')
	#Crop
#%%
for i in range(len(folderpaths)):
	print(folderpaths[i])
	PosvtArray = ito.openlistnp(folderpaths[i]+'correlationdata.npy')[0][:,0]
	stackedges = ito.openlistnp(folderpaths[i]+'edgedata.npy')
	stackedges = [arr[(arr[:,1]<yanalysisc[1]) & (arr[:,1]>yanalysisc[0])] for arr in stackedges]
	#Fit the edges and extract angles and positions
	singleProps = df.edgestoproperties(stackedges,pixrange,fitfunc,fitguess)
	AnglevtArray, EndptvtArray, ParamArrat, rotateinfo = singleProps
	ito.savelistnp(folderpaths[i]+'allDropProps.npy',singleProps)
	#Reslice data to save for each file
	dropProp[i]=np.vstack((PosvtArray,EndptvtArray[:,:,0].T,EndptvtArray[:,:,1].T,AnglevtArray.T)).T
	#Save
	#fileLabel=os.path.splitext(filenames[i]) if using files
	np.save(foldernames[i]+'DropProps',dropProp[i])




#%%
folderpaths, foldernames, dropProp = foldergen()


dropProps= [np.open(i+'DropProps') for i in forldernames]

exparams = np.genfromtxt('Aug29-SISThickness2.csv', dtype=float, delimiter=',', names=True) 

indexorder=[] #Put the lists in decending order
varr=exparams[r"Speed (um/s)"]
tsteps=exparams[r"Time per frame required"]
eedistance=exparams[r"Distance (um)"]
numcycles=exparams[r"Number of periods"]

'''
tsteps = [13,2.6,.5,1.3,0.65,.5,.48]
varr = [.1,.5,10,1,2,5,8]
indexorder=[2,6,5,4,3,1,0]
'''
tsteps = [13.6,2.7,.5,1.36,0.68,.5,.48]
varr = [0.1,0.5,10,1,2,5,8]
indexorder=[2,6,5,4,3,1,0]
eedistance=650
numcycles=[3,3,3,3,3,3,2]
timebeforestop=[2*numcycles[i]*eedistance/varr[i] for i in range(len(varr))]

labelarr=['$%.1f \mu m /s$' %i for i in varr]

def tarrf(arr,tstep):
	'''
	Simply returns a time array for plotting
	'''
	return np.linspace(0,len(arr)*tstep,len(arr)) 

colorarr=plt.cm.jet(np.linspace(0,1,len(tsteps)))
timearr=[tarrf(dropProp[i][:,0],tsteps[i]) for i in range(len(tsteps))]

#%%
forceplateaudata=[None]*len(indexorder)
for i in indexorder:	
	plateaudata=planl.plateaufilter(timearr[i],dropProp[i][:,0],timebeforestop[i],smoothparams=[2,1],sdevlims=[.1,1],outlierparam=2)
	topdata=plateaudata[4][0]
	bottomdata=plateaudata[4][1]
	tmean1,tmean2,tmean2i = planl.clusteranalysis(topdata,30)
	bmean1,bmean2,bmean2i = planl.clusteranalysis(bottomdata,30)
	forceplateaudata[i] = [topdata,tmean1,tmean2,bottomdata,bmean1,bmean2]


gs = gridspec.GridSpec(3, 1)

fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0]) 
for i in indexorder:
	ax1.plot(timearr[i]*varr[i],dropProp[i][:,0],label=labelarr[i],color=colorarr[i])
	ax1.plot(forceplateaudata[i][3][:,0]*varr[i],forceplateaudata[i][3][:,1],'k.',markersize=3)
	ax1.plot(forceplateaudata[i][0][:,0]*varr[i],forceplateaudata[i][0][:,1],'k.',markersize=3)
	ax2.plot(timearr[i]*varr[i],planl.anglefilter(dropProp[i][:,2]-dropProp[i][:,1]),color=colorarr[i])
	ax3.plot(timearr[i]*varr[i],planl.anglefilter(dropProp[i][:,5]),color=colorarr[i])
	ax3.plot(timearr[i]*varr[i],planl.anglefilter(dropProp[i][:,6]),color=colorarr[i])
	

ax1.legend()
ax1.set_ylabel('Pipette x (cc)')

ax2.set_ylabel('Droplet length (pixels)')

ax3.set_ylim(50,95)
ax3.set_ylabel('Contact angle')
ax3.set_xlabel('Approx Substrate distance travelled')

plt.tight_layout()
#%%
forceav=np.array([(i[1][0]-i[4][0])/2 for i in forceplateaudata])
errbars=np.array([np.sqrt((i[1][1]+i[4][1]))/2 for i in forceplateaudata])
plt.errorbar(varr,forceav,yerr=errbars,fmt='.')
plt.xlabel(r"Speed ($\mu m/s$")
plt.ylabel(r"Force ($px$)")
#%%


arrnum=1
testvals=planl.plateaufilter(timearr[arrnum],dropProp[arrnum][:,0],timebeforestop[arrnum],smoothparams=[50,3],sdevlims=[.1,1],outlierparam=2)
plt.plot(testvals[1])
plt.plot(testvals[2]*100)

#%%
plt.plot(timearr[arrnum],dropProp[arrnum][:,0])
plt.plot(timearr[arrnum][:815],testvals[0].T)
#%%
#%%
	
	
arrnum=-1

gs = gridspec.GridSpec(3, 1)

fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])

testv1=planl.plateaufilter(timearr[arrnum],dropProp[arrnum][:,0],timebeforestop[arrnum],smoothparams=[50,3],sdevlims=[.1,1],outlierparam=2)

ax1.plot(timearr[arrnum]*varr[arrnum],dropProp[arrnum][:,0],label='data')
ax1.plot(timearr[arrnum]*varr[arrnum],planl.smoothingfilter(dropProp[arrnum][:,0]),label='smoothed')
ax1.plot(testv1[-1][0][:,0]*varr[arrnum],testv1[-1][0][:,1],'g.',markersize=3,label='Plateau Find')
ax1.plot(testv1[-1][1][:,0]*varr[arrnum],testv1[-1][1][:,1],'g.',markersize=3)
ax1.legend()
velLim=0.2*np.std(testv1[2])
accLim=0.2*np.std(testv1[3])

ax1.set_ylabel('Force')
ax2.plot(testv1[0]*varr[arrnum],testv1[2])
ax2.axhline(velLim,c='r')
ax2.axhline(-velLim,c='r')

ax2.set_ylabel('Force\'')


ax3.plot(testv1[0]*varr[arrnum],testv1[3]*1000)
ax3.set_ylabel('Force\" (1000s)')
ax3.axhline(accLim*1000,c='r')
ax3.axhline(-accLim*1000,c='r')

xend=4300
ax1.set_xlim(0,xend)
ax2.set_xlim(0,xend)
ax3.set_xlim(0,xend)

plt.tight_layout()
#%%

#%%
gs = gridspec.GridSpec(2, 1)
fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(timearr[4]*varr[4],dropProp[4][:,2]-dropProp[4][:,0],'g-')
ax1.set_ylabel('left edge x (pixels)')

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(timearr[4]*varr[4],dropProp[4][:,3]-dropProp[4][0,3],'g-')
ax2.set_ylabel('left edge y (pixels)')
ax2.set_xlabel('Approx Substrate distance travelled')

#%%
speeds=np.array([10,5,1,.5])
displacements=np.array([30.81,28.93,23.19,19.76])
plt.plot(speeds,displacements,'.')
plt.xlabel('speed ($\mu m /s$)')
plt.ylabel('force(approx)')
#%%
'''
Scratch and Testing
Things below this are to debug bits of code etc
'''
'''

stackedges = ito.openlistnp(folderpaths[-2]+'edgedata.npy')
cropedges=[arr[(arr[:,1]<yanalysisc[1]) & (arr[:,1]>yanalysisc[0])] for arr in stackedges]


thetatorotate, leftedge = df.thetdet(cropedges)
#%%
imnum=448
rotedges=df.xflipandcombine(df.rotator(cropedges[imnum],-thetatorotate,*leftedge),leftedge[1])
plt.plot(cropedges[imnum][:,0],cropedges[imnum][:,1],'.')
plt.plot(rotedges[:,0],rotedges[:,1],'.')
fitl=df.datafitter(rotedges,False,pixrange,1,fitfunc,fitguess)


appproxsplity=np.mean(rotedges[:,0])
trimDat = rotedges[rotedges[:,0] > appproxsplity]
contactx=trimDat[np.argmin(trimDat[:,1]),0]
allcens = np.argwhere(trimDat[:,0] == contactx)
contacty = np.mean(trimDat[allcens,1])
plt.plot(contactx,contacty,'ro')


#%%
plt.plot(cropedges[imnum][:,1],cropedges[imnum][:,0],'.')

x=cropedges[imnum][:,1]
y=cropedges[imnum][:,0]
x=x[y>700]
y=y[y>700]
popt,pcov=curve_fit(df.pol2ndorder,x,y)
plt.plot(x,df.pol2ndorder(x,*popt))
#%%
endptarrayTest=np.zeros([len(cropedges),2],dtype=float)
for i in range(len(cropedges)):
	endptarrayTest[i]=df.contactptfind(cropedges[i],False,buff=0,doublesided=True)
#%%
plt.plot(endptarrayTest[:,0])
	
#%%
try2 = df.edgestoproperties(cropedges,pixrange,fitfunc,fitguess)
#%%

plt.plot(cropedges[95][:,0],cropedges[95][:,1],'.')
#%%
np.argmax(cropedges[100][:,0])
'''