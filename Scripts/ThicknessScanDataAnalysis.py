'''
This code performs the edge location and cross correlation analysis across multiple images
'''

import sys, os, glob
import matplotlib.pyplot as plt
import numpy as np
import importlib
from scipy.optimize import curve_fit
import numpy_indexed as npi

#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"F:\DualAngles2"


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


heightpaths = glob.glob(os.path.join(dataDR, "*", ""))
#%%
filenam = [None]*len(heightpaths)
velvals = [None]*len(heightpaths)
dropProp = [None]*len(heightpaths)
exparams = [None]*len(heightpaths)
for i in range(len(heightpaths)):
	folderpaths, foldernames, temp = ito.foldergen(heightpaths[i])
	datloc = os.path.join(heightpaths[i],'MainDropParams.npy')
	filenam[i], velvals[i], dropProp[i] = ito.openlistnp(datloc)
	paramloc = os.path.join(heightpaths[i],'runinfo.csv')
	exparams[i] = np.genfromtxt(paramloc, dtype=float, delimiter=',', names=True) 
#%%




springc = 0.024 #N/m
top_mperpix = 0.448e-6 #meters per pixel
side_mperpix = 0.224e-6 #meters per pixel

def extractForceandPerim(dropmainparams,fshift):
	tVals = dropmainparams[0]
	forceDat=dropmainparams[2]-fshift
	perimDat=dropmainparams[-2]
	forceplateaudata=planl.plateaufilter(tVals,forceDat,[0,tVals[-1]],smoothparams=[5,2],sdevlims=[.3,.4],outlierparam=2)	
	topidx, botidx = forceplateaudata[-1]
	meanF = (np.mean(forceDat[topidx])-np.mean(forceDat[botidx]))/2
	indexArrs = [topidx, botidx]
	
	comboind = np.logical_or(topidx,botidx)	
	perimoutmask = planl.rejectoutliers(perimDat,m=1)
	comboind = np.logical_and(comboind,perimoutmask)
	meanPerim = np.mean(perimDat[comboind])
	return meanF*springc*side_mperpix, meanPerim*top_mperpix, tVals, indexArrs

def grouper(x,y):
	'''
	Assumed sorted by speed
	'''
	result = npi.group_by(x).mean(y)
	sdev = npi.group_by(x).std(y)
	return result, sdev


meanF = np.zeros([len(sizepaths),3])
meanPerim = np.zeros([len(sizepaths),3])
tVals=[[None]*3]*len(sizepaths)
indexArr=[[None]*3]*len(sizepaths)
for i in range(len(sizepaths)):
	for j in range(3):
		res = extractForceandPerim(dropProp[i][j],np.mean(dropProp[i][0][2]))
		meanF[i,j], meanPerim[i,j], tVals[i][j], indexArr[i][j] = res


#%%
testidx=6
tVals = dropProp[testidx][0]
forceDat=dropProp[testidx][2]-fshift
#topidx, botidx = indexArrs[testidx]
plt.plot(tVals,forceDat)
plt.plot(tVals[topidx],forceDat[topidx],'r.')
plt.plot(tVals[botidx],forceDat[botidx],'r.')
#%%
plt.plot(dropProp[0][-2])
plt.ylim(660,720)

#%%



forcecombo = grouper(velvals,meanF/meanPerim[0])
perimcombo = grouper(velvals,meanPerim)
normforcecombo = grouper(velvals,meanF)

def velfit(x,B):
	return B*x**(.22)
samplex=np.linspace(0,np.max(velvals),100)
pfit,perr = curve_fit(velfit,normforcecombo[0][0],normforcecombo[0][1],sigma=normforcecombo[1][1])

print(pfit,perr)

#plt.errorbar(forcecombo[0][0],forcecombo[0][1],yerr=forcecombo[1][1],color='red',marker='.',linestyle = "None",label='Divided by constant')
plt.errorbar(normforcecombo[0][0],normforcecombo[0][1],normforcecombo[1][1],color='green',marker='.',linestyle = "None",label='Divided by perimeters')
plt.plot(samplex,velfit(samplex,*pfit))
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('speed (um/s)')
plt.ylabel('force/perimeter (arb units)')
plt.legend()
plt.tight_layout()

print(np.sqrt(np.diag(perr)))
#%%


