'''
This code performs the edge location and cross correlation analysis across multiple images
'''

import sys, os, glob, pickle
import matplotlib.pyplot as plt
import numpy as np
import importlib
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
#%%
#import similaritymeasures

#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where the folder of interest is
dataDR=r"E:/SpeedScan/"
foldername="/5umreturn_1/"

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
yanhigh=748
yanalysisc=[yanlow-y1c,yanhigh-y1c]

croppedbase=ede.cropper(noforce,*croppoints)
croppedex1=ede.cropper(ex1,*croppoints)
croppedex2=ede.cropper(ex2,*croppoints)

gs = gridspec.GridSpec(1, 3)
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
clinyguess = 214 #Guess at the center line (helpful if parts of pipette are further than droplet)
pixrange=[60,25] #xy bounding box to use in fit
#Specify an image to use as a background (needs same dim as images being analysed)
#Or can set to False
background=False 


testedge=ede.edgedetector(croppedbase,background,*imaparam)
fig = plt.figure(figsize=(8,4))
plt.imshow(croppedbase,cmap=plt.cm.gray)
plt.plot(testedge[:,0],testedge[:,1],'b.',markersize=1)
croppedforfit=testedge[(testedge[:,1]<yanalysisc[1]) & (testedge[:,1]>yanalysisc[0])]
testfit = df.datafitter(croppedforfit,True,pixrange,1,fitfunc,fitguess)
xvals=np.arange(0,20)
yvals=df.pol2ndorder(xvals,*testfit[-2])
plt.plot(xvals+testfit[0],yvals+testfit[1],'r-')
#%%
specfolder="E:/SpeedScan/5umreturn_1/"
allimages=ito.folderstackimport(specfolder)
allimages=ede.cropper(allimages,*croppoints)
#%%
noshift=croppedbase
#Find the cross correlation xvt and save to position arrays
xvals , allcorr = crco.xvtfinder(allimages,noshift,cutpoint,guassfitl)

np.save(dataDR+foldername+'CCorcents.npy',xvals)
np.save(dataDR+foldername+'CCorall.npy',allcorr)

#%%
stackedges = ede.seriesedgedetect(allimages,background,*imaparam)
ito.savelistnp(os.path.join(specfolder,'edgedata.npy'),stackedges)
#Fit the edges and extract angles and positions
#%%
stackedgecrop = [arr[(arr[:,1]<yanalysisc[1]) & (arr[:,1]>yanalysisc[0])] for arr in stackedges]
dropprops = df.edgestoproperties(stackedgecrop,pixrange,fitfunc,fitguess)
AnglevtArray, EndptvtArray, ParamArrat, rotateinfo = dropprops
ito.savelistnp(os.path.join(specfolder,'fitparams.npy'), dropprops)

#%%
plt.plot(stackedges[150][:,0],stackedges[150][:,1],'.',markersize=1)
plt.plot(stackedgecrop[150][:,0],stackedgecrop[150][:,1],'.')
rotedges=df.xflipandcombine(df.rotator(stackedges[150],-.007,0,217))
plt.plot(rotedges[:,0],rotedges[:,1],'.')
#%%
plt.plot(AnglevtArray[:,0])