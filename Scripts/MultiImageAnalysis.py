'''
This code performs the edge location and cross correlation analysis across multiple images
'''

import sys, os, glob
import matplotlib.pyplot as plt
import numpy as np
import importlib
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import pickle
#%%
#import similaritymeasures

#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"E:\SpeedScan"


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
x1c=616
x2c=1500
y1c=500
y2c=855
croppoints=[x1c,x2c,y1c,y2c]

#Select crop region for fitting (just needs to be large enough so droplet end is the max)
yanlow=679
yanhigh=758
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
cutpoint=50 # y pixel to use for cross correlation
guassfitl=20 # Number of data points to each side to use for guass fit

#Edge detection
imaparam=[-40,20,.05] #[threshval,obsSize,cannysigma]
fitfunc=df.pol2ndorder #function ie def(x,a,b) to fit to find properties
fitguess=[0,1,1]
pixrange=[30,30,25] #first two are xy bounding box for fit, last is where to search for droplet tip
#Specify an image to use as a background (needs same dim as images being analysed)
#Or can set to False
background=False 

threshtest=ede.edgedetector(croppedbase,background,*imaparam)
plt.imshow(croppedbase,cmap=plt.cm.gray)
plt.plot(threshtest[:,0],threshtest[:,1],'r.',markersize=1)
plt.axhline(cutpoint,ls='--')

#%%
'''
Run on all of the images
'''
#Import images
#Use glob to get foldernames, tif sequences should be inside
folderpaths=glob.glob(os.getcwd()+'/*/')
foldernames=next(os.walk('.'))[1]

#filenames=glob.glob("*.tif") #If using single files

#Empty array for the position vs velocity information
dropProp=[None]*len(folderpaths)
#%%
#Edge detection and save
for i in range(len(folderpaths)):
	imagestack=ito.folderstackimport(folderpaths[i])
	croppedimages=ito.cropper(imagestack,*croppoints)
	noshift=croppedbase
	#Find the cross correlation xvt and save to position arrays
	xvals , allcorr = crco.xvtfinder(croppedimages,noshift,cutpoint,guassfitl)
	ito.savelistnp(folderpaths[i]+'correlationdata.npy',[xvals,allcorr])
	
	#Define no shift cropped image as first frame, could change easily if needed
	#Perform edge detection to get python array
	stackedges = ede.seriesedgedetect(croppedimages,background,*imaparam)
	ito.savelistnp(folderpaths[i]+'edgedata.npy',stackedges) #Save for later use
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
imnum=158
stackedges = ito.openlistnp(folderpaths[1]+'edgedata.npy')
cropedges=[arr[(arr[:,1]<yanalysisc[1]) & (arr[:,1]>yanalysisc[0])] for arr in stackedges]


thetatorotate, leftedge = df.thetdet(cropedges)

rotedges=df.xflipandcombine(df.rotator(cropedges[imnum],-thetatorotate,*leftedge))
plt.plot(rotedges[:,0],rotedges[:,1],'.')
fitl=df.datafitter(rotedges,False,[60,25],1,fitfunc,fitguess)
#%%
try2 = df.edgestoproperties(cropedges,pixrange,fitfunc,fitguess)
#%%

plt.plot(cropedges[158][:,0],cropedges[158][:,1],'.')
#%%
np.argmax(cropedges[100][:,0])
#%%
np.mean(1)
#%%
def tarrf(arr,tstep):
	'''
	Simply returns a time array for plotting
	'''
	return np.linspace(0,len(arr)*tstep,len(arr)) 

tsteps=[4,0.5,2,1,1]
varr=[0.5,10,1,2,5]
labelarr=['$%.1f \mu m /s$' %i for i in varr]
indexorder=[1,4,2,0]
colorarr=['m','b','g','orange','r']
timearr=[tarrf(dropProp[i][:,0],tsteps[i]) for i in range(len(tsteps))]

#%%
from scipy.signal import savgol_filter
#%%
def anglefilter(data):
	return savgol_filter(np.abs(data),21,3)

gs = gridspec.GridSpec(3, 1)

fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0]) 
for i in indexorder:
	ax1.plot(timearr[i]*varr[i],dropProp[i][:,0],label=labelarr[i],color=colorarr[i])
	ax2.plot(timearr[i]*varr[i],dropProp[i][:,2]-dropProp[i][:,1],color=colorarr[i])
	ax3.plot(timearr[i]*varr[i],anglefilter(dropProp[i][:,5]),color=colorarr[i])
	ax3.plot(timearr[i]*varr[i],anglefilter(dropProp[i][:,6]),color=colorarr[i])
	

ax1.legend()
ax1.set_ylabel('Pipette x (cc)')

ax2.set_ylabel('Droplet length (pixels)')

ax3.set_ylim(35,90)
ax3.set_ylabel('Contact angle')
ax3.set_xlabel('Approx Substrate distance travelled')

plt.tight_layout()

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
