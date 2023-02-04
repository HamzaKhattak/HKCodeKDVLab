'''
This is a simple plotting script for the force size data
'''

import sys, os, glob, pickle, re
import matplotlib.pyplot as plt
import numpy as np
import importlib
from scipy.optimize import curve_fit
from matplotlib_scalebar.scalebar import ScaleBar
#import similaritymeasures
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"E:\PDMS\ThinPDMSDropSize"


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
rawsizescan=np.genfromtxt("SizeScan.csv",skip_header=1,delimiter=',')

def linfx2(x,a):
	return a*x
x=rawsizescan[:,2]/2
y=rawsizescan[:,9]
erdat=rawsizescan[:,10]
plt.figure(figsize=(4,3))
plt.errorbar(x,y,yerr=erdat,marker='.',linestyle="None",color='blue')
plt.xlabel("Droplet Radius ($\mathrm{\mu m}$)")
plt.ylabel("Force ($\mathrm{\mu N}$)")
pf,px = curve_fit(linfx2,x,y)
xrange=np.arange(0,np.max(x)+5)
plt.plot(xrange,linfx2(xrange,*pf),color='royalblue',linestyle='--')

file_path=r'C:\Users\WORKSTATION\Dropbox\FigTransfer\Feb17'
file_path=os.path.join(file_path,'forceradius.png')
plt.tight_layout()
plt.savefig(file_path,dpi=600)