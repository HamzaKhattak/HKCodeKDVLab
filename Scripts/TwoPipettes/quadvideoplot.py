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
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

pixsize = 1.78e-6 #pixel size of camera in m
#pixsize = 2.25e-6
'''
insecperframe = float(input('Seconds per frame in input video \n'))
xrealtime = float(input('How many times realtime to play video \n'))
savename = input('Name for video save incluing .mp4 \n')
'''
insecperframe = .25
xrealtime = 5
savename = 'multiimage.mp4'

inFPS = 1/insecperframe
outputFPS = inFPS*xrealtime

filepaths = ['run_1_MMStack_Pos0.ome.tif',
		   'run_4_MMStack_Pos0.ome.tif',
		   'run_2_MMStack_Pos0.ome.tif',
		   'run_8_MMStack_Pos0.ome.tif'
	]

quadimages = [None]*4
for i in range(4):
	quadimages[i] = imread2(filepaths[i])

aspect = quadimages[0].shape[1]/quadimages[0].shape[2]

width = 8
#%%

lets=['a','b','c','d']
nothing = np.zeros(quadimages[0][0].shape,dtype=np.uint8)
# ax refers to the axis propertis of the figure
fig, axes = plt.subplots(2,2,figsize=(width,width*aspect))
im = [None]*4
scalebars =[None]*4
for i, ax in enumerate(axes.flat):
	im[i] = ax.imshow(quadimages[i][0],cmap=plt.cm.gray,aspect='equal')
	scalebars[i] = ScaleBar(pixsize,frameon=False,location='lower right',font_properties={'size':12},pad=1.5) # 1 pixel = 0.2 meter
	
	im[i].set_clim(0, 256) #if want to reproduce original image rather than full scale
	
	ax.axis('off')
	ax.get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
	ax.get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis
	ax.axis('off')
	ax.margins(0,0)
	ax.add_artist(scalebars[i])
	ax.text(0.05, 0.95, r'$\textsf{\textbf{%s}}$' % lets[i], transform=ax.transAxes, fontweight='bold', va='top', ha='right',fontsize=12)


fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
	            hspace = 0, wspace = 0)


def init():
	"""
	This function gets passed to FuncAnimation.
	It initializes the plot axes
	"""
	#Set plot limits etc
	for i, ax in enumerate(axes.flat):
		ax.add_artist(scalebars[i])
		ax.text(0.05, 0.95, r'$\textsf{\textbf{%s}}$' % lets[i], transform=ax.transAxes, fontweight='bold', va='top', ha='right',fontsize=12)
	#plt.tight_layout()
	return im[0],im[1],im[2],im[3],
#fig.tight_layout(pad=0)

def update_plot(it):
	#Plot of image
	for i in range(4):
		if it<len(quadimages[i]):
			im[i].set_data(quadimages[i][it])
		else:
			im[i].set_data(nothing)
	return im[0],im[1],im[2],im[3],
#plt.tight_layout()

#Can control which parts are animated with the frames, interval is the speed of the animation
# now run the loop
ani = animation.FuncAnimation(fig, update_plot, frames=np.arange(0,len(quadimages[1])), interval=insecperframe/1000/xrealtime,
                    init_func=init, repeat_delay=1000, blit=True)


#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

plt.show()
#%%

Writer = animation.writers['ffmpeg']
writer = Writer(fps=outputFPS,extra_args=['-vcodec', 'libx264'])
ani.save(savename,writer=writer,dpi=200)

del ani
del Writer
del writer
