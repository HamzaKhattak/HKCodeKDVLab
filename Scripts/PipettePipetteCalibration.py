# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:06:12 2019

@author: WORKSTATION
"""

import sys, os
import matplotlib.pyplot as plt
import numpy as np
import importlib
from scipy.optimize import curve_fit
import pandas as pd
import imageio

#import similaritymeasures

#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"F:\ThinnerFilms\Calibration\lefttoright"


os.chdir(CodeDR) #Set  current working direcotry to the code directory


sys.path.append('./Tools') #Add the tools to the system path so modules can be imported

#Import required modules
import Crosscorrelation as crco
importlib.reload(crco)
import ImportTools as ito 
importlib.reload(ito)

#Remove to avoid cluttering path
sys.path.remove('./Tools') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)

#%%
#Get the image path names
displacelist = pd.read_csv('positions.csv', delimiter = ',',header = 0,dtype=object)
displacelist=displacelist[5:]
imagenames = displacelist['Image'].tolist()
positions = displacelist['Position_mm'].to_numpy(dtype=float)/1000

images=[None]*len(imagenames)
for i in range(len(images)):
	images[i]=imageio.imread(imagenames[i]+'.tiff')
#%%
#Get the image path names
k0=0.166 #N/m for the calibration pipette
metersperpixel = .2245e-6 

images=np.array(images)
base = images[0]
#%%
nwts=images.shape[0]
samplex=[None]*nwts
sampleallcorr=[None]*nwts

calibx=[None]*nwts
caliballcorr=[None]*nwts
guassfitl=20 # Number of data points to each side to use for guass fit

fig = plt.figure('Pick the y cut value for the new pipette')
plt.imshow(base,cmap='gray')
plt.imshow(images[-1],cmap='gray',alpha=0.5)
crop = (np.floor(plt.ginput(2)))
ycut=int(crop[0,1])
ycut2=int(crop[1,1])
for i in range(nwts):
	samplex[i] , sampleallcorr[i] = crco.xvtfinder(images[i],base,ycut,guassfitl)
	calibx[i] , caliballcorr[i] = crco.xvtfinder(images[i],base,ycut2,guassfitl)
	
	

#%%
samplex2 = np.abs(np.array([arr[0][0] for arr in samplex])*metersperpixel)
calibx2 = np.abs(np.array([arr[0][0] for arr in calibx])*metersperpixel)

positiondeltas = np.abs(np.abs(positions[:]-positions[0])-np.abs((calibx2[:]-calibx2[0])))
force=k0*positiondeltas

plt.figure(figsize=(4,3))
plt.plot(force*1e6,samplex2*1e6,'.')
plt.ylabel('Deflection ($\mathrm{\mu m}$)')
plt.xlabel('Force $(\mathrm{\mu N})$')

def linefx(x,a):
	return a*x

poptcalib, pcovcalib = curve_fit(linefx,force,samplex2)
poptcalibf, pcovcalibf = curve_fit(linefx,samplex2,force)

xlin=np.linspace(0,np.max(force),100)
plt.plot(xlin*1e6,linefx(xlin,*poptcalib)*1e6,label='Spring constant: %.0f $\mathrm{nN / \mu m}$' %(poptcalibf[0]*1000))
plt.legend()
plt.tight_layout()
file_path=r'E:\Calibration'
file_path=os.path.join(file_path,'PipetteCalibrationv2.png')
plt.savefig(file_path,dpi=900)
#%%
plt.plot(positiondeltas*1e6,calibx2*1e6)