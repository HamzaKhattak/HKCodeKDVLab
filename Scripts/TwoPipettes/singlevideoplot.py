# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 16:43:46 2023

@author: WORKSTATION
"""

"""
Created on Thu Nov 10 14:36:22 2022

@author: WORKSTATION
"""
import numpy as np
from skimage.io import imread as imread2
import matplotlib.pyplot as plt


from matplotlib import animation
from matplotlib_scalebar.scalebar import ScaleBar
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename


insecperframe = float(input('Seconds per frame in input video'))
inFPS = 1/insecperframe

xrealtime = float(input('How many times realtime to play video'))

outputFPS = inFPS*xrealtime

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filepath = askopenfilename() # show an "Open" dialog box and return the path to the selected file

allimages=imread2(filepath)
#allimages=allimages[:,:,::-1] #for horizontal flip
plt.imshow(allimages[0])
#%%
pixsize = 1.78e-6 #pixel size of camera in m
#pixsize = 2.25e-6
# ax refers to the axis propertis of the figure
fig, ax = plt.subplots(1,1,figsize=(8,6))
im = ax.imshow(allimages[0],cmap=plt.cm.gray,aspect='equal')
scalebar = ScaleBar(pixsize,frameon=False,location='lower right') # 1 pixel = 0.2 meter

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
	#Plot of image
	im.set_data(allimages[it])
	
	return im,
plt.tight_layout()

#Can control which parts are animated with the frames, interval is the speed of the animation
# now run the loop
ani = animation.FuncAnimation(fig, update_plot, frames=np.arange(0,len(allimages)), interval=insecperframe/1000/xrealtime,
                    init_func=init, repeat_delay=1000, blit=True)


#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

plt.show()
#%%

Writer = animation.writers['ffmpeg']
writer = Writer(fps=outputFPS,extra_args=['-vcodec', 'libx264'])
ani.save('g.mp4',writer=writer,dpi=200)

del ani
del Writer
del writer