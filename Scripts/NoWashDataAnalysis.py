'''
This code performs the edge location and cross correlation analysis across multiple images
'''

import sys, os
import matplotlib.pyplot as plt
import numpy as np
import importlib
from scipy.optimize import curve_fit
import numpy_indexed as npi

#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"F:\PDMSmigration\Thickwashed"


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

def tsplitter(s):
	def splt2(x):
		return ito.split_at(x,c='e')
	return ito.namevelfind(s, splitfunction=splt2,numLoc=-1) 

folderpaths, foldernames, dropProp = ito.foldergen(os.getcwd(),splitfunc=tsplitter)

filenam, dropProp = ito.openlistnp('MainDropParams.npy') 

'''
#In case there is a big issue with one on the speeds
velvals = np.delete(velvals,6)
filenam=np.delete(filenam,6)
del(dropProp[6])
'''
indexArrs = [None]*len(foldernames) #Empty list to store the plateau indices
exparams = np.genfromtxt('runinfo.csv', dtype=float, delimiter=',', names=True) 
timevals = np.genfromtxt('runtimes.csv', dtype=float, delimiter=',') 

#springc = 0.024 #N/m
#sidemperpix = 0.224e-6 #meters per pixel
#topmperpix = 0.448e-6 #meters per pixel

'''
want array of force, average angles, perimeter and area as final result
want where the indexes that are used for the averaging
'''
fshift = np.mean(dropProp[0][2])
meanF = np.zeros(len(foldernames))
meanPerim = np.zeros(len(foldernames))
smoothedforces=[None]*(len(foldernames))
for i in range(len(foldernames)):
	#print(i)
	tVals = dropProp[i][0]
	forceDat=dropProp[i][2]-fshift
	perimDat=dropProp[i][-2]
	forceplateaudata=planl.plateaufilter(tVals,forceDat,[0,tVals[-1]],smoothparams=[5],sdevlims=[.1,.3],outlierparam=1)	
	topidx, botidx = forceplateaudata[-1]
	smoothedforces[i] = forceplateaudata[1]
	meanF[i] = (np.mean(smoothedforces[i][topidx])-np.mean(smoothedforces[i][botidx]))/2
	indexArrs[i] = [topidx, botidx]
	comboind = np.logical_or(topidx,botidx)	
	perimoutmask = planl.rejectoutliers(perimDat,m=1)
	comboind = np.logical_and(comboind,perimoutmask)
	meanPerim[i] = np.mean(perimDat[comboind])

testidx=1
tVals = dropProp[testidx][0]
forceDat=dropProp[testidx][2]-fshift
topidx, botidx = indexArrs[testidx]
plt.plot(tVals,forceDat,'k.')
plt.plot(tVals[topidx],forceDat[topidx],'r.')
plt.plot(tVals[botidx],forceDat[botidx],'r.')

testidx=-1
tVals = dropProp[testidx][0]
forceDat=dropProp[testidx][2]-fshift
topidx, botidx = indexArrs[testidx]
plt.plot(tVals,forceDat,'g.')
plt.plot(tVals[topidx],forceDat[topidx],'r.')
plt.plot(tVals[botidx],forceDat[botidx],'r.')

#%%
perx=meanF/meanPerim
plt.plot(timevals/60/60,perx/perx[2],'.')
plt.xlabel('time(hrs)')
plt.ylabel('Force (arb)')
plt.ylim(0,1.1)
#%%
plt.plot(timevals/60/60,meanPerim,'.')
plt.xlabel('time(hrs)')
plt.ylabel('Perim (arb)')

#%%
tosave=np.array([timevals,meanF,meanPerim])
np.save('exp5.npy',tosave)
#%%
plt.plot(forceDat[0])