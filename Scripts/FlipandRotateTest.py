import sys, os
import matplotlib.pyplot as plt
import numpy as np
import importlib

#%%
import similaritymeasures
#%%
#Specify the location of the Tools folder
CodeDR="F:\TrentDrive\Research\KDVLabCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR="F:\TrentDrive\Research\Droplet forces film gradients\SlideData2"


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
allimages=ito.stackimport(dataDR+"\Translate1ums5xob.tif")
#%%
testimage1=allimages[0]
testimage2=allimages[-1]
plt.figure()
plt.imshow(testimage1)
plt.figure()
plt.imshow(testimage2)
#%%
croptest1=ede.cropper(testimage1,9,750,715,898)
croptest2=ede.cropper(testimage2,9,750,715,898)
plt.figure()
plt.imshow(croptest1)
plt.figure()
plt.imshow(croptest2)

#%%
background=np.zeros(croptest1.shape)
edges1=ede.edgedetector(croptest1,background,-100,20,.05)
edges2=ede.edgedetector(croptest2,background,-100,20,.05)
import matplotlib.colors as mcolors
colors = [(0,1,0,c) for c in np.linspace(0,1,100)]
cmapg = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=5)
plt.imshow(croptest2,cmap=plt.cm.gray)
plt.plot(edges2[:,0],edges2[:,1],'r.',markersize=1)
#%%
fitresults=df.datafitter(edges1,True,30,1,df.pol3rdorder,[0,1,1,1])

x=np.linspace(0,30,100)
plt.imshow(croptest1,cmap=plt.cm.gray)
plt.plot(x+fitresults[0],df.pol3rdorder(x,*fitresults[2])+fitresults[1],'ro',markersize=1)

#%%
#Finding how to rotate the image and flip it
combodat=np.concatenate([edges1,edges2])
plt.plot(combodat[:,0],combodat[:,1],'.')
plt.axes().set_aspect('equal')
#%%
leftedge=combodat[np.argmin(combodat[:,0])]
rightedge=combodat[np.argmax(combodat[:,0])]
print([leftedge,rightedge])
#%%
thettorot=df.angledet(*leftedge,*rightedge)
print(thettorot*180/np.pi)

#%%
#Rotate opposite the visible angle
rotationtest=df.rotator(edges2,-thettorot,*leftedge)
rotationtest=rotationtest-rotationtest[np.argmin(rotationtest[:,0])]
plt.plot(rotationtest[:,0],rotationtest[:,1],'.')
plt.axes().set_aspect('equal')

#%%
#Flip the negative values
topvalues=rotationtest[rotationtest[:,1]>0]
bottomvalues=rotationtest[rotationtest[:,1]<0]*[1,-1]
plt.plot(topvalues[:,0],topvalues[:,1],'r.')
shiftbot=bottomvalues+[0,2]
plt.plot(shiftbot[:,0],shiftbot[:,1],'b.')
plt.axes().set_aspect('equal')

#%%
thet=np.zeros([allimages.shape[0],2])
croppedimages=ede.cropper(allimages,9,750,715,898)
for i in range(allimages.shape[0]):
    edges=ede.edgedetector(croppedimages[i],background,-100,20,.05)
    rotatededges=df.rotator(edges,-thettorot,*leftedge)
    rotatededges=rotatededges-rotatededges[np.argmin(rotatededges[:,0])]
    fit=df.datafitter(rotatededges,True,30,1,df.pol3rdorder,[0,1,1,1])
    fit2=df.datafitter(rotatededges,False,30,1,df.pol3rdorder,[0,1,1,1])
    thet[i]=[fit[3],fit2[3]]
    
#%%
plt.plot(thet[:,0],label='left')
plt.plot(-thet[:,1])
plt.xlabel('frame')
plt.ylabel('angle')
plt.legend()
#%%
#testsimilarity=similaritymeasures.area_between_two_curves(bottomvalues,bottomvalues)
#testsimilarity2=similaritymeasures.area_between_two_curves(topvalues,bottomvalues)
shiftarray=np.arange(-5,5)
simarray=np.zeros(shiftarray.size)
for i in range(len(simarray)):
    simarray[i]=similaritymeasures.area_between_two_curves(topvalues,bottomvalues+[0,shiftarray[i]])

plt.plot(shiftarray,simarray/np.max(simarray))
plt.xlabel('shift (pixels)')
plt.ylabel('area')
    