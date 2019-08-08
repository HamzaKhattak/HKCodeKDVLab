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
#Check on one image
noforce=ito.imread2(dataDR+'\\base.tif')
plt.imshow(noforce)
#%%

#Specify parameters
#Cropping
#Select the minimum (1s) and maximum (2s) crop locations
x1c=828
x2c=1334
y1c=500
y2c=855
croppoints=[x1c,x2c,y1c,y2c]

croppedbase=ede.cropper(noforce,*croppoints)
plt.imshow(croppedbase)
#%%

#Cross correlation
cutpoint=50 # y pixel to use for cross correlation
guassfitl=20 # Number of data points to each side to use for guass fit

#Edge detection
imaparam=[-40,20,.05] #[threshval,obsSize,cannysigma]
fitfunc=df.pol2ndorder #function ie def(x,a,b) to fit to find properties
fitguess=[0,1,1]
pixrange=[60,60] #xy bounding box to use in fit
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

for i in range(len(folderpaths)):
    imagestack=ito.folderstackimport(folderpaths[i])
    croppedimages=ede.cropper(imagestack,*croppoints)
    #Define no shift cropped image as first frame, could change easily if needed
    noshift=croppedbase
    #Find the cross correlation xvt and save to position arrays
    xvals , allcorr=crco.xvtfinder(croppedimages,noshift,cutpoint,guassfitl)
    PosvtArray = xvals[:,0]
    #Perform edge detection to get python array
    stackedges = ede.seriesedgedetect(croppedimages,background,*imaparam)
    #Fit the edges and extract angles and positions
    AnglevtArray, EndptvtArray = df.edgestoproperties(stackedges,pixrange,fitfunc,fitguess)
    #Reslice data to save for each file
    dropProp[i]=np.vstack((PosvtArray,EndptvtArray.T,AnglevtArray.T)).T
    #Save
    #fileLabel=os.path.splitext(filenames[i]) if using files
    np.save(foldernames[i]+'DropProps',dropProp[i])
    
    
#%%
gs = gridspec.GridSpec(3, 1)

fig = plt.figure(figsize=(4,8))
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(dropProp[0][:,0],'b-',label=r'$1 \mu m /s$')
ax1.plot(dropProp[1][:,0],'r-',label=r'$5 \mu m /s$')
ax1.legend()
ax1.set_ylabel('Pipette x (cc)')

ax2 = fig.add_subplot(gs[1, 0]) 
ax2.plot(dropProp[0][:,1],'b-',label='left')
ax2.plot(dropProp[0][:,2],'b--',label='right')
ax2.plot(dropProp[1][:,1],'r-')
ax2.plot(dropProp[1][:,2],'r--')
ax2.set_ylabel('Droplet x')
ax2.legend()

ax3 = fig.add_subplot(gs[2, 0]) 
ax3.plot(dropProp[0][:,3],'b-')
ax3.plot(-dropProp[0][:,4],'b--')
ax3.plot(dropProp[1][:,3],'r-')
ax3.plot(-dropProp[1][:,4],'r--')
ax3.set_ylabel('Contact angle')
ax3.set_xlabel('Time (s)')

plt.tight_layout()

