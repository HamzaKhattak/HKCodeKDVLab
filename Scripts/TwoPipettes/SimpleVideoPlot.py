# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 21:55:54 2023

@author: WORKSTATION
"""

import pims
import numpy as np


from matplotlib import animation
from matplotlib_scalebar.scalebar import ScaleBar

import matplotlib.pyplot as plt
allimages = pims.PyAVReaderIndexed('maintwist.avi')
pixsize = 2.24e-6
insecperframe = .2
xrealtime = 5
savename = 'twistvid.mp4'

inFPS = 1/insecperframe
outputFPS = inFPS*xrealtime

dim = allimages.shape[1:]
dimr = dim[1]/dim[0]
# ax refers to the axis propertis of the figure
fig, ax = plt.subplots(1,1,figsize=(8,8/dimr))
im = ax.imshow(allimages[0],cmap=plt.cm.gray,aspect='equal')
im.set_clim(0, 256) #if want to reproduce original image rather than full scale
scalebar = ScaleBar(pixsize,frameon=False,location='lower right',font_properties={'size':26},pad=1.5) # 1 pixel = 0.2 meter
#im.set_clim(0, 256) #if want to reproduce original image rather than full scale


ax.axis('off')
ax.get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
ax.get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis







def init():
	"""
	This function gets passed to FuncAnimation.
	It initializes the plot axes
	"""
	#Set plot limits etc
	ax.add_artist(scalebar)


	#plt.tight_layout()
	return im,
#fig.tight_layout(pad=0)

def update_plot(it):
	#global xAnim, yAnim
	#Plot of image
	im.set_data(allimages[it])
	
	return im,
plt.tight_layout()

ani = animation.FuncAnimation(fig, update_plot, frames=np.arange(0,len(allimages)), interval=insecperframe/1000/xrealtime,
                    init_func=init, repeat_delay=1000, blit=True)


#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

plt.show()
#%%

Writer = animation.writers['ffmpeg']
writer = Writer(fps=outputFPS,extra_args=['-vcodec', 'libx264'])
ani.save(savename,writer=writer,dpi=200)

