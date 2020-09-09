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
from matplotlib_scalebar.scalebar import ScaleBar
#import similaritymeasures

#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"E:\DualAngles\AnalysisTesting"


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

#Some code to get the cropping worked out for the run
#now probably makes sense to have gui rather than old method
#Maybe just provide image series and index of extrema

#get the folder names and a place to store the droplet properties
folderpaths, foldernames, dropProp = ito.foldergen(os.getcwd())


noforce=ito.imread2(dataDR+'\\base.tif') #Need a no force image to compare rest of results to

#User cropping etc
#Get the images that will be used
cropselectfolder='10p0ums0'
cropindices=[50,150]
#importimage
extremapath = ito.getimpath(cropselectfolder)

extreme1 = ito.singlesliceimport(extremapath,cropindices[0])
extreme2 = ito.singlesliceimport(extremapath,cropindices[1])

#Side view cropping and selections
fig = plt.figure('Pick top left and bottom right corner and then fit lines')
plt.imshow(extreme1[0],cmap='gray')
plt.imshow(extreme2[0],cmap='gray',alpha=0.5)

print('Select crop points')
cropside = np.floor(plt.ginput(2)) #format is [[xmin,ymin],[xmax,ymax]]
print('Select contact angle analysis limit')

#Get analysis limit points
climit = np.floor(plt.ginput(2))
climit = [climit[0,1],climit[1,1]]

yanalysisc=[climit[0]-cropside[0,1],climit[1]-cropside[0,1]]

plt.close(fig)


#Top view cropping and selections
fig = plt.figure('Pick top left and bottom right corner')

plt.imshow(extreme1[1],cmap='gray')
plt.imshow(extreme2[1],cmap='gray',alpha=0.5)
croptop = np.floor(plt.ginput(2))

plt.close(fig)


#%%

testing=ito.cropper2(extreme2[1],croptop)
plt.imshow(testing)
#%%%





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
gs = gridspec.GridSpec(1, 3)
fig = plt.figure(figsize=(8,4))
#plt.imshow(extreme2[0],cmap=plt.cm.gray)

croppedbase=ito.cropper2(noforce,cropside)
croppedex1=ito.cropper2(extreme1[0],cropside)
croppedex2=ito.cropper2(extreme2[0],cropside)

ax1 = fig.add_subplot(gs[0, 0])

ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax2.axhline(yanalysisc[0])
ax2.axhline(yanalysisc[1])
ax1.imshow(croppedbase,cmap=plt.cm.gray)
ax2.imshow(croppedex1,cmap=plt.cm.gray)
ax3.imshow(croppedex2,cmap=plt.cm.gray)
#%%

#Cross correlation
cutpoint=10 # y pixel to use for cross correlation
guassfitl=20 # Number of data points to each side to use for guass fit

#Edge detection
imaparam=[np.min(croppedex1)/.7,20,.05] #[threshval,obsSize,cannysigma]
fitfunc=df.pol2ndorder #function ie def(x,a,b) to fit to find properties
fitguess=[0,1,1]
pixrange=[60,60,25] #first two are xy bounding box for fit, last is where to search for droplet tip
#Specify an image to use as a background (needs same dim as images being analysed)
#Or can set to False
background=False 

threshtest=ede.edgedetector(croppedex1,background,*imaparam)

fig,ax = plt.subplots(1)
ax.imshow(croppedex1,cmap=plt.cm.gray)
ax.plot(threshtest[:,0],threshtest[:,1],'r.',markersize=1)
ax.axhline(cutpoint,ls='--')

ax.axis('off')


scalebar = ScaleBar(0.75e-6,frameon=False,location='upper right') # 1 pixel = 0.2 meter
ax.add_artist(scalebar)


#%%
folderpaths, foldernames, dropProp = ito.foldergen(os.getcwd())
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
	np.save(folderpaths[i]+'DropProps',dropProp[i])

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