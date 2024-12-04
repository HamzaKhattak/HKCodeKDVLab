# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:06:12 2019

@author: WORKSTATION
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import imageio as io
from scipy.optimize import curve_fit

import tifffile as tf
import cv2 as cv
from pyometiff import OMETIFFReader
#import similaritymeasures

#%%

#Specify where the data is and where plots will be saved
dataDR=r"F:\ShaganaFilms\5ksample3\stretch1_1"

filepath = r"F:\ShaganaFilms\5ksample3\stretch1_1\stretch1_MMStack_Pos0.ome.tif"

zerobase = True
basepath = r"F:\ShaganaFilms\ControlSample1\base.tif"

savename = '5KS2R1'

#Set working directory to data location
os.chdir(dataDR)

reader = OMETIFFReader(fpath=filepath)
img_array, metadata, xml_metadata = reader.read()

if zerobase:
	base = img_array[0]
else:
	base = img_array[0]
#%%
points =img_array[:,0,0]
plt.plot(points)
#%%
endpoint=800
plt.imshow(img_array[endpoint])

from scipy import signal

def ccor(a,b,meth='cv.TM_CCOEFF_NORMED'):
	'''
	This code runs cross correlation on an image with a given template
	'''
	#Normalize the input vectors
	norma = (a - np.mean(a)) / (np.std(a))
	normb = (b - np.mean(b)) / (np.std(b))
	w, h = a.shape[::-1]
	match = signal.correlate(norma, normb, mode='full', method='auto')
	match=match/np.max(match)
	#Getting shifts
	#corrx = np.arange(2*norma.shape[1]-1)-(norma.shape[1]-1)
	#corry = np.arange(2*norma.shape[0]-1)-(norma.shape[0]-1)
	shiftx = norma.shape[1]-1
	shifty = norma.shape[0]-1
	return shiftx,shifty,match

def ccor2(a,b):
	'''
	This code runs cross correlation on an image with a given template
	'''
	#Normalize the input vectors
	match = cv.matchTemplate(a,b,method=cv.TM_CCOEFF_NORMED)
	#Getting shifts
	#corrx = np.arange(2*norma.shape[1]-1)-(norma.shape[1]-1)
	#corry = np.arange(2*norma.shape[0]-1)-(norma.shape[0]-1)
	shiftx = a.shape[1]-1
	shifty = a.shape[0]-1
	return shiftx,shifty,match

#%%
plt.imshow(img_array[0])
plt.title('input first the force pipette y section and then the position pipette')
initialcrops = plt.ginput(4)

initialcrops = np.array(initialcrops).astype(int)
plt.close()

forceims = img_array[:endpoint,initialcrops[0][1]:initialcrops[1][1]]

posims = img_array[:endpoint,initialcrops[2][1]:initialcrops[3][1]]
fig, ax = plt.subplots(2,2)
ax[0,0].imshow(forceims[0])
ax[0,1].imshow(posims[0])
ax[1,0].imshow(forceims[-1])
ax[1,1].imshow(posims[-1])
#%%
plt.imshow(base)

plt.title('input first the force pipette template and then the position pipette')
templatecrops = plt.ginput(4)
templatecrops = np.array(templatecrops).astype(int)

templateF = base[templatecrops[0][1]:templatecrops[1][1],templatecrops[0][0]:templatecrops[1][0]]


templateP = base[templatecrops[2][1]:templatecrops[3][1],templatecrops[2][0]:templatecrops[3][0]]
plt.close()

fig, ax = plt.subplots(2,1)
ax[0].imshow(templateP)
ax[1].imshow(templateF)

#%%
forcex = np.zeros(endpoint)
posx = np.zeros(endpoint)
for i in range(endpoint):
	xs,ys,matchF = ccor2(forceims[i],templateF)
	forcex[i] = np.argmax(matchF[0])
	
	xs,ys,matchp = ccor(posims[i],templateP)
	maxloc = np.unravel_index(matchp.argmax(), matchp.shape)
	posx[i] = maxloc[1]

np.save(savename+'.npy',[posx,forcex])


#%%
plt.plot(posx-posx[0],'.',label='pos')
plt.plot(forcex-forcex[0],'.',label='force')
plt.legend()
#%%
#Get the image path names



from matplotlib import animation




# ax refers to the axis propertis of the figure
fig, ax = plt.subplots(1,1,figsize=(8,6))
im = ax.imshow(img_array[0],cmap=plt.cm.gray,aspect='equal')

ax.axis('off')
ax.get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
ax.get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis


track1, = ax.plot([posx[0],0],color='red',animated=True)
track2, = ax.plot([posx[0],0],color='cyan',animated=True)

def update_plot(it):
	#global xAnim, yAnim
	#Plot of image
	im.set_data(img_array[it])
	track1.set_data([[posx[it]-50,posx[it]-50],[0,800]])
	track2.set_data([[forcex[it]+120,forcex[it]+190],[1400,2200]])
	return im,track1,track2,
plt.tight_layout()

#Can control which parts are animated with the frames, interval is the speed of the animation
# now run the loop
ani = animation.FuncAnimation(fig, update_plot, frames=np.arange(0,endpoint,10), interval=1,
                     repeat_delay=1000, blit=True)


#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

plt.show()


#%%
k0=2.58 #N/m for the calibration pipette
mppix = .4504e-6
width = 7.2e-3
length = 12.7e-3
thickness = 19e-6

xpull=(posx-posx[0])*mppix
xsense = (forcex-forcex[0])*mppix
Fs = xsense*k0


stress = Fs/(width*thickness)
strain = (xpull-xsense)/length
stress=stress[strain>0]
strain = strain[strain>0]

cstress = stress[50:-100]
cstrain = strain[50:-100]
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
plt.plot(cstrain*100,cstress/1000,'.')
plt.xlabel('$e \ (\%)$')
plt.ylabel('$\sigma$ (kPa)')
def linfx(x,a,b):
	return a*x

popt,perr = curve_fit(linfx,cstrain,cstress)
plt.plot(cstrain*100,linfx(cstrain, *popt)/1000)
print(popt/1e6)
plt.xlim(0,)
plt.ylim(0,)