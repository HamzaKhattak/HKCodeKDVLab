'''
This code combines the results of the previous programs
'''

import sys, os, glob, pickle, re
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
dataDR=r"E:\PDMS\Compare"


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
Thinnest=np.load('PDMSEvenThinnerpvveldat.npy')
Thinner=np.load('PDMSThinnerpvveldat.npy')
Thick=np.load('SpeedScanpvveldat.npy')
IonicThick=np.load('IonicThickpvveldat.npy')
IonicThin=np.load('IonicRunpvveldat.npy')
IonicThinner=np.load('IonicThinpvveldat.npy')
IonicIntermediate=np.load('IonicIntermediatepvveldat.npy')

#%%
def fitfunc(x,a):
	return a*x**(1/4)
def errorbarplot(dataset,labeld,cval,normval=1):
	#normval=np.mean(dataset[dataset[:,0]>8,1])
	ax.errorbar(dataset[:,0],dataset[:,1]/normval,yerr=dataset[:,2]/normval,marker='.',linestyle='None',line=None,Label=labeld,color=cval)
	fit,pcov = curve_fit(fitfunc,dataset[:,0],dataset[:,1]/normval,maxfev=100000)
	xlist=np.linspace(0.1,10,100)
	ax.plot(xlist,fitfunc(xlist,*fit),linestyle='dashed',color=cval)
	print(fit)
	
file_path=r'C:\Users\WORKSTATION\Dropbox\FigTransfer\Feb17'
file_path=os.path.join(file_path,'ComboForces.png')
fig,ax=plt.subplots(figsize=(5,4));

#errorbarplot(Thick,"Thick",'k')
#errorbarplot(Thinner,"Thinner",)
#errorbarplot(Thinnest,"Thinnest")

errorbarplot(IonicThick,"Ionic Thick",'b',normval=328/(0.75/2))
errorbarplot(IonicIntermediate,"Ionic Intermediate",'r',normval=290/(0.75/2))
errorbarplot(IonicThinner,"Ionic Thin",'g',normval=285/(0.75/2))
#ax.set_xscale("log")
#ax.set_yscale("log")
ax.legend(loc='lower right')
ax.set_xlabel('Speed ($\mathrm{\mu m /s}$)')
ax.set_ylabel(r'Force ($\mathrm{\mu N}$)')
fig.tight_layout()
#%%
def linfx(x,a,b):
	return a*x+b
logdatx=np.log(Thinnest[:,0])
logdaty=np.log(Thinnest[:,1])
fity,fitp = curve_fit(linfx,logdatx,logdaty)
print(fity)
print(np.sqrt(np.diag(fitp)))
plt.plot(logdatx,logdaty,'.')
plt.plot(logdatx,linfx(logdatx,*fity))
#%%
max1=np.max(Thick[:,1])
max2=np.max(Thinner[:,1])
max3=np.max(Thinnest[:,1])
fig,ax=plt.subplots(figsize=(5,4))
ax.errorbar(Thick[:,0],Thick[:,1]/max1,'.',yerr=Thick[:,2]/max1,Label='Thick')
ax.errorbar(Thinner[:,0],Thinner[:,1]/max2,'.',yerr=Thinner[:,2]/max2,Label='Thin')
ax.errorbar(Thinnest[:,0],Thinnest[:,1]/max3,'.',yerr=Thinnest[:,2]/max3,Label='Thinnest')
ax.errorbar(IonicThick[:,0],IonicThick[:,1]/max3,'.',yerr=IonicThick[:,2]/max3,Label='Ionic Thick')
ax.errorbar(IonicThin[:,0],IonicThin[:,1]/max4,'.',yerr=IonicThin[:,2]/max4,Label='Ionic Thin')

ax.legend(loc='lower right')
ax.set_xlabel('Speed ($\mathrm{\mu m /s}$)')
ax.set_ylabel(r'Force ($\mathrm{\mu N}$)')
fig.tight_layout()
plt.savefig(file_path,dpi=900)

#%%
