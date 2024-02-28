# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 21:55:54 2023

@author: WORKSTATION
"""

import pims, os
import numpy as np


from matplotlib import animation
from matplotlib_scalebar.scalebar import ScaleBar
import tifffile as tf
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",'font.size': 9,
})

#allimages = pims.PyAVReaderIndexed('maintwist.avi')
imdir = r'F:\ferro\Experiments\Concentration05\Pip3\multidrop4_1'
imname = 'multidrop4_MMStack_Pos0.ome.tif'
savedir = r'G:\My Drive\LabShare\2024\APStransfer'
savename = 'initialdrops.mp4'
os.chdir(imdir)

tifobj = tf.TiffFile(imname)
numFrames = len(tifobj.pages)
allimages =  tf.imread(imname,key=slice(0,numFrames))

#allimages=allimages[:,300:-300,300:-300] #crop as needed

pixsize = 2.24e-6 #pixel size for pulse and 2x
#pixsize = .24e-6 #pixsize for 20x objective on ace python 5000 camera
insecperframe = .25
xrealtime = 10


inFPS = 1/insecperframe
outputFPS = inFPS*xrealtime

dim = allimages.shape[1:]
dimr = dim[1]/dim[0]
# ax refers to the axis propertis of the figure
fig, ax = plt.subplots(1,1,figsize=(4,4/dimr))
im = ax.imshow(allimages[0],cmap=plt.cm.gray,aspect='equal')
im.set_clim(0, 256) #if want to reproduce original image rather than full scale
#scalebar = ScaleBar(pixsize,frameon=False,location='upper left',font_properties={'size':24},pad=1.5)
scalebar = ScaleBar(pixsize,frameon=False,location='upper left',font_properties={'size':24},pad=1.5,label_formatter = lambda x, y:'') 
ax.add_artist(scalebar)

ax.axis('off')
ax.get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
ax.get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis


fig.subplots




def init():
	"""
	This function gets passed to FuncAnimation.
	It initializes the plot axes
	"""
	#Set plot limits etc
	


	#plt.tight_layout()
	return im,
#fig.tight_layout(pad=0)

def update_plot(it):
	#global xAnim, yAnim
	#Plot of image
	im.set_data(allimages[it])
	
	return im,
plt.tight_layout()

fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

ani = animation.FuncAnimation(fig, update_plot, frames=np.arange(0,len(allimages)), interval=insecperframe/1000/xrealtime, repeat_delay=1000, blit=True)


#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

plt.show()
#%%
os.chdir(savedir)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=outputFPS,extra_args=['-vcodec', 'libx264'])
ani.save(savename,writer=writer,dpi=200)

