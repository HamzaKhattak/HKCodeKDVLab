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
dataDR=r"E:\Calibration\Calib\calibpipette"


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
weightlist = pd.read_csv('weightlist.csv', delimiter = ',',header = 0,dtype=object)
imagenames = weightlist['Name'].tolist()
weights = weightlist['weight-grams'].to_numpy(dtype=float)/1000
#%%
images=[None]*len(imagenames)
for i in range(len(images)):
	images[i]=np.rot90(imageio.imread(imagenames[i]))
images=np.array(images)
base = images[0]
plt.imshow(base)
#%%
nwts=images.shape[0]
crop=np.zeros([nwts,3,2])
xvals=[None]*nwts
allcorr=[None]*nwts
guassfitl=20 # Number of data points to each side to use for guass fit
for i in range(nwts):
	fig = plt.figure('Pick top left and bottom right corner')
	plt.imshow(base,cmap='gray')
	plt.imshow(images[i],cmap='gray',alpha=0.5)
	crop[i]= (np.floor(plt.ginput(3)))
	plt.close(fig)
	x1 = int(crop[i,0,0])
	x2 = int(crop[i,1,0])
	ycut=int(crop[i,2,1])
	cropwt=images[i,:,x1:x2]
	cropbase=base[:,x1:x2]
	xvals[i] , allcorr[i] = crco.xvtfinder(cropwt,cropbase,ycut,guassfitl)
	

#%%
force=9.81*weights
metersperpixel = .2245e-6 

xval2=np.array([arr[0][0] for arr in xvals])*metersperpixel
plt.figure(figsize=(4,3))
plt.plot(force*1e6,xval2*1e6,'.')
plt.ylabel('Deflection ($\mathrm{\mu m}$)')
plt.xlabel('Force $(\mathrm{\mu N})$')

def linefx(x,a):
	return a*x

poptcalib, pcovcalib = curve_fit(linefx,force,xval2)
poptcalibf, pcovcalibf = curve_fit(linefx,xval2,force)

xlin=np.linspace(0,np.max(force),100)
plt.plot(xlin*1e6,linefx(xlin,*poptcalib)*1e6,label='Spring constant: %.0f $\mathrm{nN / \mu m}$' %(poptcalibf[0]*1000))
plt.legend()
plt.tight_layout()
file_path=r'E:\Calibration'
file_path=os.path.join(file_path,'PipetteCalibrationv2.png')
plt.savefig(file_path,dpi=900)