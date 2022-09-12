# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 21:29:36 2022

@author: WORKSTATION
"""


'''
This code performs the edge location and cross correlation analysis across multiple images
'''

import sys, os
import matplotlib.pyplot as plt
import numpy as np
import importlib
from scipy.optimize import curve_fit
import numpy_indexed as npi
import imageio
#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"F:\PDMSMigration\decay\decaying_1"


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

#Remove to avoid cluttering path
sys.path.remove('./Tools') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)
#%%
infile='decaying_1_MMStack_Default.ome.tif'

#imageframes=ito.fullseqimport(os.path.join(dataDR,infile))
imageframes = ito.folderimport(dataDR,'.tif')
#%%
imageframes = np.array(imageframes)
#%%
imageframes = imageframes.astype(np.int8)
#%%
subtractim=imageframes[400]
maxval = np.max(subtractim) 

imav=np.mean(imageframes,axis=(1,2))
overallmean = np.mean(imav)
brightdiff = imav-overallmean
maxbrightdiff=np.max(brightdiff)


normed=np.array(imageframes-subtractim+maxval,dtype='uint8')

numFrames = normed.shape[0]
#%%

plt.imshow(normed[0],cmap='gray')
#%%
plt.plot(imav,'.')
#%%
testim = normed[0]
centerfindarray=np.argwhere(testim>200)

eggparam = df.eggfitter(centerfindarray[:,0],centerfindarray[:,1])
cx,cy = df.contourfinder(centerfindarray[:,0],centerfindarray[:,1],eggparam[0])
plt.imshow(testim,cmap='gray')
plt.plot(cy,cx)
plt.plot(np.mean(cy),np.mean(cx),'o')

centerx=np.mean(cx)
centery=np.mean(cy)
#%%

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile 

#%%
radialarraytest = radial_profile(normed[0], [centery,centerx])
radialarraylength = radialarraytest.shape[0]
profilesarray = np.zeros([numFrames,radialarraylength])
for i in range(numFrames):
	profilesarray[i] = radial_profile(normed[i], [centery,centerx])

#%%
plt.plot(profilesarray[10])
plt.plot(profilesarray[500])

#%%
"""
Matplotlib Animation Example

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 800), ylim=(-20, 60))
line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    x = np.arange(radialarraylength)
    y = profilesarray[i]-np.mean(profilesarray[i,600:700])
    line.set_data(x, y)
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=1000, interval=20, blit=True)
#%%
# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
anim.save('radialprofiletrack.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()





#%%
outFile1 = 'test.mp4'
imageio.mimwrite(outFile1, normed ,quality=10, input_params=['-r','10'],  output_params=['-r', '30'])