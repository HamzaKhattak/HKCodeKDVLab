# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:24:23 2024

@author: WORKSTATION
"""

import numpy as np
import matplotlib.pyplot as plt



import imageio, os, importlib, sys, time


from matplotlib import colors
from win11toast import notify
import tifffile as tf

import requests
#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"F:\ferro\TwoDrops\2percentrun2_4"

#Use telegram to notify
tokenloc = r"F:\ferro\token.txt"


os.chdir(CodeDR) #Set  current working direcotry to the code directory


sys.path.append('./Scripts/FerroFluidTrack') #Add the tools to the system path so modules can be imported

#Import required modules
import PointFindFunctions as pff
importlib.reload(pff)

import FrametoTimeAndField as fieldfind
importlib.reload(fieldfind)

import NNfindFunctions as nnfind
importlib.reload(nnfind)

#Remove to avoid cluttering path
sys.path.remove('./Scripts/FerroFluidTrack') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)

#%%
imfile = '2percentrun2_MMStack_Pos0.ome.tif'
Guassinputfile = 'p2s2r4.csv'
savename ='twodroponly.mp4'



fieldfind.findGuassVals(Guassinputfile, imfile,'FrameGaussVals')
tifobj = tf.TiffFile(imfile)
numFrames = len(tifobj.pages)
ims =  tf.imread(imfile,key=slice(0,numFrames))
ims = ims[:,450:-350,550:-500]

import matplotlib.patches as patches
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",'font.size': 16,
})

import matplotlib.animation as animation


from matplotlib_scalebar.scalebar import ScaleBar


gaussvals = np.loadtxt('FrameGaussVals.csv',delimiter=',')[:,2]
maxgauss = np.max(gaussvals)

pixsize = 2.24e-6
insecperframe = .5
xrealtime = 10
skipframe = 1
inFPS = 1/insecperframe
outputFPS = inFPS*xrealtime


dim = ims.shape[1:]
dimr = dim[1]/dim[0]
# ax refers to the axis propertis of the figure
fig, ax = plt.subplots(1,1,figsize=(8,8/dimr))

im = ax.imshow(ims[0],cmap='gray')
im.set_clim(0,180)
txt = ax.text(.8, .95, '$B={x:.1f} \, \mathrm{{G}}$'.format(x=gaussvals[0]), ha='center',transform=plt.gca().transAxes)
rect = patches.Rectangle((409, 25), 100*gaussvals[0]/maxgauss, 12, linewidth=1, edgecolor='r', facecolor='red')
ax.add_patch(rect)
scalebar = ScaleBar(pixsize,frameon=False,location='upper left',pad=0.5) 
ax.add_artist(scalebar)
ax.axis('off')

ax.get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
ax.get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis

fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)



def animate(i):
	im.set_data(ims[i])
	rect.set_width(100*gaussvals[i]/maxgauss)
	#rect.set_width(100)
	txt.set_text('$B={x:.1f} \, \mathrm{{G}}$'.format(x=gaussvals[i]))
	return im,txt,rect,

ani = animation.FuncAnimation(fig, animate, frames=np.arange(0,len(ims),skipframe), interval=skipframe*insecperframe/1000/xrealtime,
                     repeat_delay=1000, blit=True)


#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
print(len(ims)/outputFPS)
plt.show()

#%%
Writer = animation.writers['ffmpeg']
writer = Writer(fps=outputFPS/skipframe,extra_args=['-vcodec', 'libx264'])
ani.save(savename,writer=writer,dpi=200)