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
import tkinter as tk
from tkinter import filedialog
import numpy_indexed as npi



#import similaritymeasures

#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"E:\DualAngles\SecondSpeedScan"


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

filenam, velvals, dropProp = ito.openlistnp('MainDropParams.npy') 

indexArrs = [None]*len(velvals) #Empty list to store the plateau indices
exparams = np.genfromtxt('runinfo.csv', dtype=float, delimiter=',', names=True) 

springc = 0.155 #N/m
mperpix = 0.75e-6 #meters per pixel

'''
want array of force, average angles, perimeter and area as final result
want where the indexes that are used for the averaging
'''
fshift = np.mean(dropProp[0][2])
meanF=np.zeros(len(velvals))
meanPerim=np.zeros(len(velvals))
for i in range(len(velvals)):
	tVals = dropProp[i][0]
	forceDat=dropProp[i][2]-fshift
	perimDat=dropProp[i][-2]
	forceplateaudata=planl.plateaufilter(tVals,forceDat,[0,tVals[-1]],smoothparams=[6,1],sdevlims=[0.5,.5],outlierparam=2)	
	topidx, botidx = forceplateaudata[-1]
	meanF[i] = (np.mean(forceDat[topidx])-np.mean(forceDat[botidx]))/2
	indexArrs[i] = [topidx, botidx]
	
	comboind = np.logical_or(topidx,botidx)	
	perimoutmask = planl.rejectoutliers(perimDat,m=1)
	comboind = np.logical_and(comboind,perimoutmask)
	meanPerim[i] = np.mean(perimDat[comboind])

#%%


testidx=20
tVals = dropProp[testidx][0]
forceDat=dropProp[testidx][2]-fshift
plt.plot(tVals,forceDat)
plt.plot(tVals[topidx],forceDat[topidx],'r.')
plt.plot(tVals[botidx],forceDat[botidx],'r.')
#%%
def grouper(x,y):
	'''
	Assumed sorted by speed
	'''
	res = npi.group_by(x).mean(y)
	sdev = npi.group_by(x).std(y)
	return res, sdev
#%%
forcecombo = grouper(velvals,meanF/meanPerim[0])
perimcombo = grouper(velvals,meanPerim)
normforcecombo = grouper(velvals,meanF/meanPerim)
#%%
#%%
plt.errorbar(forcecombo[0][0],forcecombo[0][1],yerr=forcecombo[1][1],color='red',marker='.',linestyle = "None",label='Divided by constant')
plt.errorbar(normforcecombo[0][0],normforcecombo[0][1],normforcecombo[1][1],color='green',marker='.',linestyle = "None",label='Divided by perimeters')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('speed (um/s)')
plt.ylabel('force/perimeter (arb units)')
plt.legend()
plt.tight_layout()
#%%
tArry = dropProp[0][0]
perims = dropProp[0][-2]
masktest = planl.rejectoutliers(perims,m=1)
plt.plot(tArry[masktest],perims[masktest],'.')
plt.plot(tArry,perims)
#%%

plt.plot(np.sort(velvals),meanPerim,'.')
#%%
plt.plot(np.sort(velvals),meanPerim,'.')
#%%
testidx=0
plt.plot(dropProp[testidx][0],dropProp[testidx][2])
plt.plot(dropProp[testidx+1][0],dropProp[testidx+1][2])
#%%
runName=os.path.basename(os.getcwd())

forcevt=np.column_stack([varr,forceav,errbars,anglemeans,anglestds])
np.save(runName+'pvveldat.npy',forcevt)
#%%
plt.plot(dropProp[0][:,2]-dropProp[0][:,1],label="0.1")

plt.plot(dropProp[1][:,2]-dropProp[1][:,1],'r--',label="0.2")

plt.plot(dropProp[2][:,2]-dropProp[2][:,1],label="0.5")

plt.plot(dropProp[3][:,2]-dropProp[3][:,1],label="1")
plt.legend()

