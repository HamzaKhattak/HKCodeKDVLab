# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:02:36 2024

@author: WORKSTATION
"""

import imageio, sys, os, importlib
import matplotlib.pyplot as plt
import tifffile as tf
import cv2 as cv
import numpy as np
from scipy import signal

from scipy.optimize import curve_fit

#%%

#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"F:/ferro/air/largedrops/run_7"

#Use telegram to notify
tokenloc = r"F:\ferro\token.txt"


os.chdir(CodeDR) #Set  current working direcotry to the code directory


sys.path.append('./Scripts/FerroFluidTrack') #Add the tools to the system path so modules can be imported

#Import required modules
import AirDropFunctions as adf
importlib.reload(adf)



import FrametoTimeAndField as ftf
importlib.reload(ftf)

#Remove to avoid cluttering path
sys.path.remove('./Scripts/FerroFluidTrack') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)

#%%
#Import the images of interest and a base image for background subtraction
tifobj = tf.TiffFile('run_MMStack_Pos0.ome.tif')
numFrames = len(tifobj.pages)
ims =  tf.imread('run_MMStack_Pos0.ome.tif',key=slice(0,numFrames))


#%%
newims=np.zeros((ims.shape[0],ims.shape[1]-200,ims.shape[2]-900))
dat = np.load('run3locs.npy')
xshifts = (dat[:,2]+dat[:,3])/2
xshifts = xshifts-xshifts[0]
for i, im in enumerate(ims):
	xshift = int(xshifts[i])
	print(xshift)
	newims[i] = adf.cropper(im, [[500+xshift,0],[-400+xshift,-200]])
#%%




import matplotlib.animation as animation


from matplotlib_scalebar.scalebar import ScaleBar


gaussvals = np.loadtxt('guassconvert7.csv',delimiter=',')[:,2]


mperpix = 2.25e-6
pixsize = 2.24e-6
insecperframe = .5
xrealtime = 20
savename = 'stabilforce.mp4'

inFPS = 1/insecperframe
outputFPS = inFPS*xrealtime


dim = newims.shape[1:]
dimr = dim[1]/dim[0]
# ax refers to the axis propertis of the figure
fig, ax = plt.subplots(1,1,figsize=(5,5/dimr))

im = ax.imshow(newims[0],cmap='gray')
#txt = ax.text(.2, .05, 'B={x:.1f}G'.format(x=gaussvals[0]),fontsize=12, ha='center',transform=plt.gca().transAxes)
scalebar = ScaleBar(mperpix,frameon=False,location='upper right',pad=0.5) 
#ax.add_artist(scalebar)
ax.axis('off')

ax.get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
ax.get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis

fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
def init():
	"""
	This function gets passed to FuncAnimation.
	It initializes the plot axes
	"""
	#Set plot limits etc


	#plt.tight_layout()
	return 


def animate(i):
	im.set_data(newims[i])
	#txt.set_text('B={x:.1f}G'.format(x=gaussvals[i]))
	return im,

ani = animation.FuncAnimation(fig, animate, frames=np.arange(0,len(ims)), interval=insecperframe/1000/xrealtime,
                     repeat_delay=1000, blit=True)


#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

plt.show()
#%%

Writer = animation.writers['ffmpeg']
writer = Writer(fps=outputFPS,extra_args=['-vcodec', 'libx264'])
ani.save(savename,writer=writer,dpi=200)