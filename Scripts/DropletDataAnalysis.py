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
folderpaths, foldernames, dropProp = ito.foldergen(os.getcwd())

dropProp = [np.load(i+'/DropProps.npy') for i in folderpaths]

exparams = np.genfromtxt('Aug29-SISThickness2.csv', dtype=float, delimiter=',', names=True) 
#%%
#These are all in descending order of speed, so reverse to match dropProps
varr=exparams[r"Speed_ums"][::-1]
tsteps=exparams[r"Time_per_frame_required"][::-1]
eedistance=exparams[r"Distance_um"][::-1]
numcycles=exparams[r"Number_of_periods"][::-1]
indexorder=[i for i in range(varr.size)] #If want in another order for some reason
#%%

'''
tsteps = [13,2.6,.5,1.3,0.65,.5,.48]
varr = [.1,.5,10,1,2,5,8]
indexorder=[2,6,5,4,3,1,0]
'''
timebeforestop=[2*numcycles[i]*eedistance[i]/varr[i] for i in range(len(varr))]

labelarr=['$%.1f \mu m /s$' %i for i in varr]

def tarrf(arr,tstep):
	'''
	Simply returns a time array for plotting
	'''
	return np.linspace(0,len(arr)*tstep,len(arr)) 

colorarr=plt.cm.jet(np.linspace(0,1,len(tsteps)))
timearr=[tarrf(dropProp[i][:,0],tsteps[i]) for i in range(len(tsteps))]


#%%

#%%
forceplateaudata=[None]*len(indexorder)
angleplateaudata=[None]*len(indexorder)
areanormed = [arr[:,0]/planl.anglefilter(arr[:,2]-arr[:,1]) for arr in dropProp]
for i in indexorder:	
	plateaudata=planl.plateaufilter(timearr[i],dropProp[i][:,0],[30,timebeforestop[i]],smoothparams=[2,1],sdevlims=[.1,1],outlierparam=2)
	topdata=plateaudata[4][0]
	bottomdata=plateaudata[4][1]
	idh=plateaudata[5][0]
	idl=plateaudata[5][1]
	tmean1,tmean2,tmean2i = planl.clusteranalysis(topdata,30)
	bmean1,bmean2,bmean2i = planl.clusteranalysis(bottomdata,30)
	angleplateaudata=dropProp[i][:,5][idh]
	forceplateaudata[i] = [topdata,idh,tmean1,tmean2,bottomdata,idl,bmean1,bmean2]
	angleplateaudata[i] = [topdata,idh,tmean1,tmean2,bottomdata,idl,bmean1,bmean2]
forceplateaudata[1][1]

#%%
plt.plot(timearr[0]*forceplateaudata[0][1],dropProp[0][:,0]*forceplateaudata[0][1],'.')
#%%
plt.plot(timearr[0][forceplateaudata[0][1]],'.')
#%%
gs = gridspec.GridSpec(3, 1)

fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0]) 
for i in indexorder:
	ax1.plot(timearr[i]*varr[i],dropProp[i][:,0],label=labelarr[i],color=colorarr[i])
	ax1.plot(timearr[i][forceplateaudata[i][5]]*varr[i],dropProp[i][:,0][forceplateaudata[i][5]],'k.')
	#ax1.plot(forceplateaudata[i][3][:,0]*varr[i],forceplateaudata[i][3][:,1],'k.',markersize=3)
	#ax1.plot(forceplateaudata[i][0][:,0]*varr[i],forceplateaudata[i][0][:,1],'k.',markersize=3)
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
forceav=np.array([(i[2][0]-i[6][0])/2 for i in forceplateaudata])
errbars=np.array([np.sqrt((i[1][1]**2+i[6][1]**2))/2 for i in forceplateaudata])
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