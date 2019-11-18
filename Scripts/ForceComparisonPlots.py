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
dataDR=r"E:\SoftnessTest\RunCompare"


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
PS2Run2=np.load('PS2Run2pvveldat.npy')
SpeedScanPS2=np.load('SpeedScanPS2pvveldat.npy')
SISRun1=np.load('SIS3per14wt1pvveldat.npy')
SISRun2=np.load('SISThickness2pvveldat.npy')



#%%
file_path=r'C:\Users\WORKSTATION\Dropbox\FigTransfer\Symposium Day'
file_path=os.path.join(file_path,'ForcevSpeed.png')

max1=np.max(PS2Run2[-3,1])
max2=np.max(SISRun1[:,1])
max3=np.max(SISRun2[:,1])
max4=np.max(SpeedScanPS2[:,1])
fig,ax=plt.subplots(figsize=(5,4))

ax.errorbar(PS2Run2[:,0],PS2Run2[:,1],yerr=PS2Run2[:,2],fmt='r.',Label='Stiff')
ax.set_xlabel('Speed ($\mathrm{\mu m /s}$)')
ax.set_ylabel(r'Force ($\mathrm{\mu N})$')
fig.tight_layout()
plt.savefig(file_path,dpi=900)
#%%
def fitfunc(x,a,b,c):
	return 1/(1+c*np.exp(-b*x))
xlist=np.linspace(0.1,10,100)

PSAllx=np.concatenate([PS2Run2[:,0],SpeedScanPS2[:,0]])
PSAlly=np.concatenate([PS2Run2[:,1]/max1,SpeedScanPS2[:,1]/max4])
fit1,pcov = curve_fit(fitfunc,PSAllx,PSAlly)
fit2,pcov = curve_fit(fitfunc,SISRun1[:,0],SISRun1[:,1]/max2)
fit3,pcov = curve_fit(fitfunc,SISRun2[:,0],SISRun2[:,1]/max3)


file_path=r'C:\Users\WORKSTATION\Dropbox\FigTransfer\Symposium Day'
file_path=os.path.join(file_path,'smallstiffandsoft.png')


fig,ax=plt.subplots(figsize=(4,3))
ax.errorbar(PS2Run2[:,0],PS2Run2[:,1]/max1,yerr=PS2Run2[:,2]/max1,fmt='r.',Label='Stiff')

ax.errorbar(SpeedScanPS2[:,0],SpeedScanPS2[:,1]/max4,yerr=SpeedScanPS2[:,2]/max4,fmt='r.')

ax.plot(xlist,fitfunc(xlist,*fit1),'r--')

ax.errorbar(SISRun2[:,0],SISRun2[:,1]/max3,yerr=SISRun2[:,2]/max3,fmt='g.',Label='Soft elastic')

ax.plot(xlist,fitfunc(xlist,*fit3),'g--')

#ax.errorbar(SISRun1[:,0],SISRun1[:,1]/max2,yerr=SISRun1[:,2]/max2,fmt='b.',Label='Thinner elastic')
#ax.plot(xlist,fitfunc(xlist,*fit2),'b--')
#ax.legend(loc='lower right')
ax.set_ylim(0.38,1.18)

ax.set_xlabel('Speed ($\mathrm{\mu m /s}$)')
ax.set_ylabel(r'$\frac{F}{F_o}$')
fig.tight_layout()
plt.savefig(file_path,dpi=900)
#%%

file_path=r'C:\Users\WORKSTATION\Dropbox\FigTransfer\Symposium Day'
file_path=os.path.join(file_path,'ACtForce0.png')

max1=np.max(PS2Run2[-3,1])
max2=np.max(SISRun1[:,1])
max3=np.max(SISRun2[:,1])
max4=np.max(SpeedScanPS2[:,1])
fig,ax=plt.subplots(figsize=(5,4))
ax.errorbar(PS2Run2[:,0],PS2Run2[:,1],yerr=PS2Run2[:,2],fmt='r.',Label='Stiff glassy')
ax.errorbar(SpeedScanPS2[:,0],SpeedScanPS2[:,1],yerr=SpeedScanPS2[:,2],fmt='r.')
ax.errorbar(SISRun2[:,0],SISRun2[:,1],yerr=SISRun2[:,2],fmt='g.',Label='Soft thicker')
ax.errorbar(SISRun1[:,0],SISRun1[:,1],yerr=SISRun1[:,2],fmt='b.',Label='Soft thinner')

ax.legend(loc='lower right')
ax.set_xlabel('Speed ($\mathrm{\mu m /s}$)')
ax.set_ylabel(r'Force ($\mathrm{\mu N}$)')
fig.tight_layout()
plt.savefig(file_path,dpi=900)
