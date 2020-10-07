'''
This code performs the edge location and cross correlation analysis across multiple images
'''

import sys, os, glob, pickle, re, time
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
dataDR=r"E:\DualAngles\SixthScan"


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
#%%

noforce=ito.imread2(dataDR+'\\base.tif') #Need a no force image to compare rest of results to

#User cropping etc
#Get the images that will be used
cropselectfolder='10p0ums1'
cropindices=[41,150]
#importimage
extremapath = ito.getimpath(cropselectfolder)

extreme1 = ito.singlesliceimport(extremapath,cropindices[0])
extreme2 = ito.singlesliceimport(extremapath,cropindices[1])
#%%
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
gs = gridspec.GridSpec(4, 4)
fig = plt.figure(figsize=(6,4))
#plt.imshow(extreme2[0],cmap=plt.cm.gray)


croppedbase=ito.cropper2(noforce,cropside)
croppedsidechecks=[ito.cropper2(extreme1[0],cropside),ito.cropper2(extreme2[0],cropside)]
croppedtopchecks=[ito.cropper2(extreme1[1],croptop),ito.cropper2(extreme2[1],croptop)]


pixelsize=0.75e-6

#Cross correlation
cutpoint=30 # y pixel to use for cross correlation
guassfitl=20 # Number of data points to each side to use for guass fit

#Side Edge detection
sideimaparam=[-1*np.max(croppedsidechecks[0])/3,40,.05] #[threshval,obsSize,cannysigma]
sidebackground=False 

sidethreshtest=ede.edgedetector(croppedsidechecks[0],sidebackground,*sideimaparam)

#Top Edge detection
topimaparam=[-1*np.max(croppedtopchecks[0])/1.8,60,.05] #[threshval,obsSize,cannysigma]
topbackground=False 

topthreshtest=ede.edgedetector(croppedtopchecks[0],topbackground,*topimaparam)
topthreshtest2=ede.edgedetector(croppedtopchecks[1],topbackground,*topimaparam)

ax1 = fig.add_subplot(gs[:2, :2])
ax2 = fig.add_subplot(gs[0, 2:])
ax3 = fig.add_subplot(gs[1, 2:])
ax4 = fig.add_subplot(gs[1:, :2])
ax5 = fig.add_subplot(gs[1:, 2:])






#Plotting
ax2.imshow(croppedbase,cmap=plt.cm.gray)
ax2.axis('off')

ax1.imshow(croppedsidechecks[0],cmap=plt.cm.gray)
ax1.plot(sidethreshtest[:,0],sidethreshtest[:,1],'r.',markersize=1)
ax1.axhline(yanalysisc[0])
ax1.axhline(yanalysisc[1])
ax1.axhline(cutpoint,ls='--')
ax1.axis('off')
scalebar = ScaleBar(pixelsize,frameon=False,location='lower left') # 1 pixel = 0.2 meter
ax1.add_artist(scalebar)

ax3.imshow(croppedsidechecks[1],cmap=plt.cm.gray)
ax3.axis('off')
ax3.axhline(yanalysisc[0])
ax3.axhline(yanalysisc[1])


ax4.imshow(croppedtopchecks[0],cmap=plt.cm.gray)
ax4.plot(topthreshtest[:,0],topthreshtest[:,1],'r.',markersize=1)
ax4.axis('off')
ax5.imshow(croppedtopchecks[1],cmap=plt.cm.gray)
ax5.plot(topthreshtest2[:,0],topthreshtest2[:,1],'r.',markersize=1)
ax5.axis('off')
plt.tight_layout()

#save parameters
edparams = [cropside,sideimaparam,croptop,topimaparam,yanalysisc]
ito.savelistnp('edgedetectparams.npy',edparams)

#%%
#Edge detection and save
for i in range(len(folderpaths)):
	#Import image
	t1=time.time()
	imagepath = ito.getimpath(folderpaths[i])
	imseq = ito.fullseqimport(imagepath)
	
	#Seperate out side and top views
	sidestack = imseq[:,0]
	topstack = imseq[:,1]
	
	#Complete edge detection and cross correlation for side images
	croppedsides=ito.cropper2(sidestack,cropside)
	#Cross correlation
	noshift=croppedbase
	xvals , allcorr = crco.xvtfinder(croppedsides,noshift,cutpoint,guassfitl)
	ito.savelistnp(folderpaths[i]+'correlationdata.npy',[xvals,allcorr])
	#Perform edge detection
	sideedges = ede.seriesedgedetect(croppedsides,sidebackground,*sideimaparam)
	ito.savelistnp(folderpaths[i]+'sideedgedata.npy',sideedges) #Save for later use
	
	#Complete edge detection for top view
	croppedtops=ito.cropper2(topstack,croptop)
	topedges = ede.seriesedgedetect(croppedtops,topbackground,*topimaparam)
	ito.savelistnp(folderpaths[i]+'topedgedata.npy',topedges) #Save for later use
	
	#analysis
	ttake = time.time()-t1
	print("%s completed in %d seconds." % (folderpaths[i], ttake))
#%%
'''
This code no longer involves the images and can run much faster
Also has bits that are most commonly changed (ie fitting functions etc)
This does fits etc to the output of the edge detection
'''
#open parameters
cropside,sideimaparam,croptop,topimaparam,yanalysisc = ito.openlistnp('edgedetectparams.npy')

fitfunc=df.pol2ndorder #function ie def(x,a,b) to fit to find properties
fitguess=[0,1,1]
pixrange=[120,120,10] #first two are xy bounding box for fit, last is where to search for droplet tip
#Specify an image to use as a background (needs same dim as images being analysed)
#Or can set to False
runparams = np.genfromtxt('runinfo.csv', dtype=float, delimiter=',', names=True)
runparams = np.sort(runparams,order = r"Speed_ums")
speedvals = runparams[r"Speed_ums"]/1e6 #Speed is inputted into the device

limit1vals = runparams[r"Point_1_mm"]/1e3 
limit2vals = runparams[r"Point_2_mm"]/1e3
distancevals = np.abs(limit1vals-limit2vals)
numFramevals = runparams[r"Number_of_frames"].astype(np.int)

secperframevals = 2*distancevals/speedvals/numFramevals

#labels for the useful data
labels = ['time','distance travelled','crco','langle','rangle','perimeter','area']
#for i in range(len(folderpaths)):
for i in range(len(speedvals)):
	print(folderpaths[i])
	#Side view data
	#Get correlation positions
	PosvtArray = ito.openlistnp(folderpaths[i]+'correlationdata.npy')[0][:,0]
	
	#Get contact angles etc from side profile, use cuttoffs for analysis
	sidestackedges = ito.openlistnp(folderpaths[i]+'sideedgedata.npy')
	sidestackedges = [arr[(arr[:,1]<yanalysisc[1]) & (arr[:,1]>yanalysisc[0])] for arr in sidestackedges]
	#Fit the edges and extract angles and positions
	sideProps = df.edgestoproperties(sidestackedges,pixrange,fitfunc,fitguess)
	AnglevtArray, EndptvtArray, ParamArrat, rotateinfo = sideProps
	
	#Topview
	topstackedges = ito.openlistnp(folderpaths[i]+'topedgedata.npy')
	topProps = df.seriescomboperimcalc(topstackedges)
	tparams, fitcirc, meanlocs, perims, areas = topProps
	#Save all the fit data
	ito.savelistnp(folderpaths[i]+'allDropProps.npy',[sideProps,topProps])
	#Save the useful bits
	#All the values that go into the final array
	tvals = np.linspace(0,secperframevals[0]*numFramevals[0],numFramevals[0])
	dtrav = tvals*speedvals[i]
	
	#Get all of the useful data into a single file
	dropProp[i] = np.array([tvals,dtrav,PosvtArray,AnglevtArray[:,0],AnglevtArray[:,1],perims,areas])

	#Save
	np.save(folderpaths[i]+'DropProps',dropProp[i])
	
	

ito.savelistnp('MainDropParams.npy',[foldernames,speedvals,dropProp])
#%%
sidestackedges = ito.openlistnp(folderpaths[0]+'sideedgedata.npy')
plt.plot(sidestackedges[0][:,0],sidestackedges[0][:,1],'.')