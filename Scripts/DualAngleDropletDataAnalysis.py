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
dataDR=r"F:\DualAngles2\Exp2"


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


folderpaths, foldernames, dropProp = ito.foldergen(os.getcwd())

filenam, velvals, dropProp = ito.openlistnp('MainDropParams.npy') 

'''
#In case there is a big issue with one on the speeds
velvals = np.delete(velvals,6)
filenam=np.delete(filenam,6)
del(dropProp[6])
'''
indexArrs = [None]*len(velvals) #Empty list to store the plateau indices
exparams = np.genfromtxt('runinfo.csv', dtype=float, delimiter=',', names=True) 

springc = 0.024 #N/m
mperpix = 0.224e-6 #meters per pixel

'''
want array of force, average angles, perimeter and area as final result
want where the indexes that are used for the averaging
'''
fshift = np.mean(dropProp[0][2])
meanF = np.zeros(len(velvals))
meanPerim = np.zeros(len(velvals))
smoothedforces=[None]*(len(velvals))
for i in range(len(velvals)):
	#print(i)
	tVals = dropProp[i][0]
	forceDat=dropProp[i][2]-fshift
	perimDat=dropProp[i][-2]
	forceplateaudata=planl.plateaufilter(tVals,forceDat,[0,tVals[-1]],smoothparams=[5],sdevlims=[.1,.5],outlierparam=1)	
	topidx, botidx = forceplateaudata[-1]
	smoothedforces[i] = forceplateaudata[1]
	meanF[i] = (np.mean(smoothedforces[i][topidx])-np.mean(smoothedforces[i][botidx]))/2
	indexArrs[i] = [topidx, botidx]
	comboind = np.logical_or(topidx,botidx)	
	perimoutmask = planl.rejectoutliers(perimDat,m=1)
	comboind = np.logical_and(comboind,perimoutmask)
	meanPerim[i] = np.mean(perimDat[comboind])



testidx=-1
tVals = dropProp[testidx][0]
forceDat=dropProp[testidx][2]-fshift
topidx, botidx = indexArrs[testidx]
plt.plot(tVals,forceDat,'k.')
plt.plot(tVals[topidx],forceDat[topidx],'r.')
plt.plot(tVals[botidx],forceDat[botidx],'r.')

testidx=0
tVals = dropProp[testidx][0]
forceDat=dropProp[testidx][2]-fshift
topidx, botidx = indexArrs[testidx]
plt.plot(tVals,forceDat,'k.')
plt.plot(tVals[topidx],forceDat[topidx],'r.')
plt.plot(tVals[botidx],forceDat[botidx],'r.')


#%%
def grouper(x,y):
	'''
	Assumed sorted by speed
	'''
	result = npi.group_by(x).mean(y)
	sdev = npi.group_by(x).std(y)
	return result, sdev


forcecombo = grouper(velvals,meanF)
perimcombo = grouper(velvals,meanPerim)
normforcecombo = grouper(velvals,meanF/meanPerim)

def velfit(x,B,a):
	return B*x**a
samplex=np.linspace(0,np.max(velvals),100)
pfit,perr = curve_fit(velfit,normforcecombo[0][0],normforcecombo[0][1],sigma=normforcecombo[1][1])

print(pfit)
plt.figure()
#plt.errorbar(forcecombo[0][0],forcecombo[0][1],yerr=forcecombo[1][1],color='red',marker='.',linestyle = "None",label='Divided by constant')
plt.errorbar(normforcecombo[0][0],normforcecombo[0][1],normforcecombo[1][1],color='green',marker='.',linestyle = "None",label='Divided by perimeters')
plt.plot(samplex,velfit(samplex,*pfit))
plt.xscale('log')
plt.yscale('log')
plt.xlabel('speed (um/s)')
plt.ylabel('force/perimeter (arb units)')
plt.legend()
plt.tight_layout()

print(np.sqrt(np.diag(perr)))
#np.save('Thick90Sum.npy',[normforcecombo[0][0],normforcecombo[0][1],normforcecombo[1][1]])


#%%
xlog = np.log(normforcecombo[0][0])
ylog = np.log(normforcecombo[0][1])
#errlog = np.log(normforcecombo[1][1])

def plaw(x,a,b):
	return a*x+b
plt.plot(xlog,ylog,'.')
pfit,perr = curve_fit(plaw,xlog,ylog)
print(pfit)

