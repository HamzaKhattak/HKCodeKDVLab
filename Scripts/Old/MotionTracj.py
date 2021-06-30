'''
Find movement in an image
'''

import sys, os
import importlib
import matplotlib.pyplot as plt
import numpy as np

#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved, use forward slashes
dataDR=r"E:\PDMS\evapabsorb\test_2"


os.chdir(CodeDR) #Set  current working direcotry to the code directory


sys.path.append('./Tools') #Add the tools to the system path so modules can be imported

#Import required modules
import ImportTools as ito 
importlib.reload(ito)
import ImportTools as ito 
importlib.reload(ito)
import EdgeDetection as ede
importlib.reload(ede)

#Remove to avoid cluttering path
sys.path.remove('./Tools') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)

#%%
inFile="2ums_1_MMStack_Default.ome.tif"

#Import the tif files in a folder
imageframes=ito.omestackimport(dataDR)
#Or just one file
#%%
plt.imshow(imageframes[100])
croppoints=(np.floor(plt.ginput(2)))
croppoints=croppoints.T.flatten().astype(int)
imtest=ito.cropper(imageframes[100],*croppoints)
plt.imshow(imtest)
#%%
#Edge detection
plt.imshow(imtest,cmap=plt.cm.gray)
imaparam=[-100,20,.05] #[threshval,obsSize,cannysigma]
#Specify an image to use as a background (needs same dim as images being analysed)
#Or can set to False
background=False 

threshtest=ede.edgedetector(imtest,background,*imaparam)
plt.plot(threshtest[:,0],threshtest[:,1],'g.')
comloc=np.mean(threshtest,axis=0)
plt.plot(*comloc,'ro')
lengtharr=imageframes.shape[0]

#%%
serieslength=imageframes.shape[0]
croppedimages=ito.cropper(imageframes,*croppoints)
#%%
edgevalsdust=ede.seriesedgedetect(croppedimages,background,*imaparam)
comlocs=np.zeros([serieslength,2])
for i in range (serieslength):
	comlocs[i]=np.mean(edgevalsdust[i],axis=0)

#%%
dt=300
mperpix=0.75e-6
timedat=np.linspace(0,dt*serieslength,serieslength)
xstart=comlocs[0,0]
ystart=comlocs[0,1]
xshifted=(comlocs[:,0]-xstart)*0.75e-6
yshifted=(comlocs[:,1]-ystart)*0.75e-6
RMS=np.sqrt(xshifted**2+yshifted**2)
plt.plot(timedat,xshifted)
plt.plot(timedat,yshifted)
#%%
plt.plot(timedat,RMS)
#%%
#Temperature data
envirotimes=np.loadtxt('envirodata.csv', dtype='datetime64',delimiter=',',skiprows=1, usecols=[1])
print('Start time is:',(envirotimes[0]))
deltatimes=envirotimes-envirotimes[0]
deltatimes=deltatimes/np.timedelta64(1, 's')
envirodata=np.loadtxt('envirodata.csv', delimiter=',',skiprows=1, usecols=[2,3])
syncpoint1=45 #index for time in the temp=humidity meter that matches that start of the evap
syncpoint2=344 #index for time in the temp=humidity meter that matches that end of the evap
#syncpoint2 ins't too important, just for plotting purposes

envirot=deltatimes[syncpoint1:syncpoint2]-deltatimes[syncpoint1]
envirodat=envirodata[syncpoint1:syncpoint2]
#%%
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(2, 1)

fig = plt.figure(figsize=(6,6))

ax1 = fig.add_subplot(gs[0, 0])

ax1.plot(timedat/3600,xshifted*1e6,'k-',label='x')
ax1.plot(timedat/3600,yshifted*1e6,'k--',label='y')
ax1.plot(timedat/3600,RMS*1e6,'g--',label='RMS')
ax1.set_ylabel('RMS position ($\mu m$)')
ax1.legend()

ax2 = fig.add_subplot(gs[1, 0],sharex=ax1)



color = 'tab:red'
ax2.set_xlabel('time (hrs)')
ax2.set_ylabel('Temperature $^o C$', color=color)
ax2.plot(envirot/3600,envirodat[:,0], color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax3 = ax2.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax3.set_ylabel('Humidity (%)', color=color)  # we already handled the x-label with ax1
ax3.plot(envirot/3600,envirodat[:,1], color=color)
ax3.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

plt.legend()

#%%
#plot RMS vs temperature
interT=np.interp(timedat, envirot, envirodat[:,0])
interH=np.interp(timedat, envirot, envirodat[:,1])

#%%
plt.plot( envirot, envirodat[:,0])
plt.plot( timedat, interT)
#%%
fig=plt.figure(figsize=(5,4))
plt.plot(interT,RMS*1e6,'.')
plt.xlabel('Temperature ($^o C$)')
plt.ylabel('RMS pos ($\mu m$)')
fig.tight_layout()


