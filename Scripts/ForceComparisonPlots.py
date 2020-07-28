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
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
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

Ionic1100=np.load('IonicThickpvveldat.npy')
Ionic150=np.load('IonicRunpvveldat.npy')
Ionic48=np.load('IonicThinpvveldat.npy')
Ionic68=np.load('IonicIntermediatepvveldat.npy')
Ionic90=np.load('IonicIntermediate2pvveldat.npy')

cfac=(0.75/2)*1e-6
rIonic1100=305*cfac
rIonic150=295*cfac
rIonic48=290*cfac
rIonic68=303*cfac
rIonic85=300*cfac
#ax.set_xscale("log")

#%%
plt.rc('font',size=18)
plt.rc('lines',markersize=8)
def fitfunc(x,a):
	return a*x**(1/4)
def errorbarplot(dataset,labeld,cval,normval=1):
	x=dataset[:,0]
	y=1e-6*dataset[:,1]/normval/2
	yer=1e-6*dataset[:,2]/normval/2
	#normval=np.mean(dataset[dataset[:,0]>8,1])
	ax.errorbar(x,y*1000,(yer+y*.05)*1000,marker='.',linestyle='None',line=None,Label=labeld,color=cval)
	fit,pcov = curve_fit(fitfunc,x,y,maxfev=100000)
	xlist=np.linspace(0.1,10,100)
	ax.plot(xlist,fitfunc(xlist,*fit)*1000,linestyle='dashed',color=cval)
	return fit, np.sqrt(np.diag(pcov))
	
file_path=r'C:\Users\WORKSTATION\Dropbox\FigTransfer\Feb24th'
file_path=os.path.join(file_path,'framelog5.png')

fig,ax=plt.subplots(figsize=(7,5));

fit1100=errorbarplot(Ionic1100,"1100 nm",'b',rIonic1100)
fit150=errorbarplot(Ionic150,"150 nm",'darkblue',rIonic150)
fit90=errorbarplot(Ionic90,"85 nm",'c',rIonic85)
fit68=errorbarplot(Ionic68,"68 nm",'g',rIonic68)
fit48=errorbarplot(Ionic48,"48 nm",'r',rIonic48)

ax.set_yscale("log")
ax.set_xscale("log")

ax.legend(loc='lower right')
ax.set_xlabel('Speed ($\mathrm{\mu m /s}$)')
ax.set_ylabel(r'$F/r$ ($\mathrm{mN/m}$)')
ax.set_ylim(5,140)
#ax.set_ylabel(r'Force (normalized)')

fig.tight_layout()
plt.savefig(file_path,dpi=600)
#%%
h=np.array([1100,150,68,48,85])
hol=h
const=np.array([fit1100,fit150,fit68,fit48,fit90])
consty=const[:,0].flatten()
conster=const[:,1].flatten()+0.05*consty

def pfunc(x,a,b):
	return np.piecewise(x, 
                        [x < b, x >= b],
                        [lambda x: a*x**(3/2), lambda x: a*b**(3/2)])




pfit,per = curve_fit(pfunc,hol,consty,p0=[0.6,120],sigma=conster,absolute_sigma=True)

fig=plt.figure(figsize=(7,6))
plt.errorbar(hol,consty*100,yerr=conster*100,marker='.',linestyle='None')
xrange=np.linspace(20,1200,1000)
plt.plot(xrange,pfunc(xrange,*pfit)*100)
plt.xlabel(r'$h$ (nm)')
plt.axvline(pfit[1],linestyle='--',color='orange')
plt.xscale('log')
plt.ylabel(r'$\beta/100$')
plt.tight_layout()
print(pfit)
file_path=r'C:\Users\WORKSTATION\Dropbox\FigTransfer\Feb17'
file_path=os.path.join(file_path,'Thickness.png')
plt.savefig(file_path,dpi=900,transparent=True)
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
