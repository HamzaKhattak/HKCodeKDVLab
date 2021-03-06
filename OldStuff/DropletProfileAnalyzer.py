import sys, os, glob
import matplotlib.pyplot as plt
import numpy as np
import importlib
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
#%%
import similaritymeasures

#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"E:\Newtips\SpeedAnalysis"


os.chdir(CodeDR) #Set  current working direcotry to the code directory


sys.path.append('./Tools') #Add the tools to the system path so modules can be imported

#Import required modules
import DropletprofileFitter as df
importlib.reload(df)
import ImportTools as ito 
importlib.reload(ito)
import EdgeDetection as ede
importlib.reload(ede)

#Remove to avoid cluttering path
sys.path.remove('./Tools') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)
#%%
#Import images
imagestack=ito.stackimport(dataDR+r"\1ums.tif")
#%%
#Select the minimum (1s) and maximum (2s) crop locations
x1c=300
x2c=900
y1c=400
y2c=1000
croppoints=[x1c,x2c,y1c,y2c]

fig, ax = plt.subplots(nrows=2, ncols=2)
testimage1=imagestack[0]
testimage2=imagestack[-1]


croptest1=ede.cropper(testimage1,*croppoints)
croptest2=ede.cropper(testimage2,*croppoints)

ax[0,0].imshow(testimage1)
ax[0,1].imshow(testimage2)

ax[1,0].imshow(croptest1)
ax[1,1].imshow(croptest2)

#%%
#check that edge detection is working properly

#Create a zero background or could import one and crop
background=np.zeros(croptest1.shape)


#[threshval,obsSize,cannysigma]
imaparam=[-30,20,.05]

#Have two edges
edges1=ede.edgedetector(croptest1,background,*imaparam)
edges2=ede.edgedetector(croptest2,background,*imaparam)


#Colormap to show fitting
colors = [(0,1,0,c) for c in np.linspace(0,1,100)]
cmapg = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=5)

#Plot to show
plt.imshow(croptest2,cmap=plt.cm.gray)
plt.plot(edges2[:,0],edges2[:,1],'r.',markersize=1)


#%%

#Crop and find the edges for all of the images
croppedimages=ede.cropper(imagestack,x1c,x2c,y1c,y2c)
alledges=ede.seriesedgedetect(croppedimages,background,*imaparam)

#%%

#Rotating and flipping the points for a better fit
leftlineinfo, rightlineinfo = df.linedet(alledges)
allcontactpts=np.concatenate([leftlineinfo[:500],rightlineinfo[:500]])

#Fit a line
def linfx(x,m,b):
    return m*x + b
fitlineparam,firlinecov = curve_fit(linfx,allcontactpts[:,0],allcontactpts[:,1])

xlist=np.linspace(np.min(allcontactpts[:,0]),np.max(allcontactpts[:,0]),100)


#create 2 random points based on the line for the angle detection function
leftedge=[0,linfx(0,*fitlineparam)]
rightedge=[1,linfx(1,*fitlineparam)]
thettorot=df.angledet(*leftedge,*rightedge)
print(thettorot*180/np.pi)

plt.imshow(croptest1,cmap=plt.cm.gray)
plt.imshow(croptest2,cmap=plt.cm.gray,alpha=0.3)
plt.plot(allcontactpts[:,0],allcontactpts[:,1])
plt.plot(xlist,linfx(xlist,*fitlineparam))
plt.axes().set_aspect('equal')

#%%
#Rotate opposite the visible angle
rotationtest=df.rotator(alledges[0],-thettorot,*leftedge)
#Center the result
rotationtest=rotationtest-[0,rotationtest[np.argmin(rotationtest[:,0])][1]]
#Flip the negative values
topvalues=rotationtest[rotationtest[:,1]>0]
bottomvalues=rotationtest[rotationtest[:,1]<0]*[1,-1]

shiftbot=bottomvalues+[0,0]

gs = gridspec.GridSpec(2, 3)

fig = plt.figure()
ax1 = fig.add_subplot(gs[0, :]) # row 0, span all columns
ax1.imshow(croptest1,cmap=plt.cm.gray)
ax1.imshow(croptest2,cmap=plt.cm.gray,alpha=0.3)
ax1.plot(allcontactpts[:,0],allcontactpts[:,1])
ax1.plot(xlist,linfx(xlist,*fitlineparam))
ax1.plot([0,1])
ax1.set_aspect('equal')

ax2 = fig.add_subplot(gs[1, 0]) # row 1, col 0
ax2.plot(alledges[0][:,0],alledges[0][:,1],'.')
ax2.set_aspect('equal')

ax3 = fig.add_subplot(gs[1, 1]) # row 1, col 1
ax3.plot(rotationtest[:,0],rotationtest[:,1],'.')
ax3.set_aspect('equal')


ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(topvalues[:,0],topvalues[:,1],'r.')
ax4.plot(shiftbot[:,0],shiftbot[:,1],'b.')
ax4.set_aspect('equal')
#%%
#Check the polynomial fit, the 0,1,1,1 are the guesses for the parameters 
#30 is thebuffer for which data to include in the fit
combodat=np.concatenate([topvalues,bottomvalues])
fitresults=df.datafitter(combodat,True,[60,60],1,df.pol2ndorder,[0,1,1])
x=np.linspace(0,60,100)
plt.plot(combodat[:,0],combodat[:,1],'.')
plt.plot(x+fitresults[0],df.pol2ndorder(x,*fitresults[2])+fitresults[1],'ro',markersize=1)
plt.set_aspect('equal')

#%%
testflip=df.xflipandcombine(df.rotator(alledges[0],-thettorot,*leftedge))
plt.plot(testflip[:,0],testflip[:,1],'.',markersize=1)
plt.set_aspect('equal')
#%%
teststuffang, teststuffpos = df.edgestoproperties(alledges,[60,60],df.pol2ndorder,[0,1,1])

plt.plot(teststuffpos[:,1])
plt.plot(teststuffpos[:,0])
#%%
'''
This code test shifts in y to align the data which is currently not needed.
'''
'''
#testsimilarity=similaritymeasures.area_between_two_curves(bottomvalues,bottomvalues)
#testsimilarity2=similaritymeasures.area_between_two_curves(topvalues,bottomvalues)
shiftarray=np.arange(-5,5)
simarray=np.zeros(shiftarray.size)
for i in range(len(simarray)):
    simarray[i]=similaritymeasures.area_between_two_curves(topvalues,bottomvalues+[0,shiftarray[i]])

plt.plot(shiftarray,simarray/np.max(simarray))
plt.xlabel('shift (pixels)')
plt.ylabel('area')
'''