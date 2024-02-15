# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 13:00:14 2024

@author: WORKSTATION
"""

from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.animation as animation
import FrametoTimeAndField as ftf
import AirDropFunctions as adf
import imageio
import sys
import os
import importlib
import matplotlib.pyplot as plt
import tifffile as tf
import cv2 as cv
import numpy as np
from scipy import signal

from scipy.optimize import curve_fit

# %%

# Specify the location of the Tools folder
CodeDR = r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
# Specify where the data is and where plots will be saved
dataDR = r"F:/ferro/air/largedrops/run_3"

# Use telegram to notify
tokenloc = r"F:\ferro\token.txt"


os.chdir(CodeDR)  # Set  current working direcotry to the code directory


# Add the tools to the system path so modules can be imported
sys.path.append('./Scripts/FerroFluidTrack')

# Import required modules
importlib.reload(adf)


importlib.reload(ftf)

# Remove to avoid cluttering path
sys.path.remove('./Scripts/FerroFluidTrack')  # Remove tools from path

# Set working directory to data location
os.chdir(dataDR)

# %%
# Import the images of interest and a base image for background subtraction
tifobj = tf.TiffFile('run_MMStack_Pos0.ome.tif')
numFrames = len(tifobj.pages)
ims = tf.imread('run_MMStack_Pos0.ome.tif', key=slice(0, numFrames))
# %%
plt.imshow(ims[0], cmap='gray')

'''
Find center locations of droplets initially, use threshold to get initial y values
'''
threshims = adf.findthresh(ims[0], 1, threshold=30)
cents, contours = adf.twodropsxyfind(threshims)


lefty0 = cents[0][1]
righty0 = cents[1][1]

# %%
'''
Find the initial x position using pipette location
'''

# Crop the left and right pipettes
leftcrop = [[600, 200], [750, 500]]
rightcrop = [[910, 200], [1110, 500]]
left = adf.cropper(ims, leftcrop)
right = adf.cropper(ims, rightcrop)


ileftline = adf.findshifted(ims[:2], leftcrop)
irightline = adf.findshifted(ims[:2], rightcrop)

initialcenterleft = adf.findcents(ileftline[1][0], ileftline[0], lefty0)
initialcenterright = adf.findcents(irightline[1][0], irightline[0], righty0)


plt.plot(contours[1][:, 0, 0], contours[1][:, 0, 1], '.')
plt.plot(cents[0][0], cents[0][1], 'ro')
plt.plot(cents[1][0], cents[1][1], 'ro')

plt.plot(initialcenterleft[1], initialcenterleft[0], 'go')
plt.plot(initialcenterright[1], initialcenterright[0], 'go')


plt.imshow(ims[0], cmap='gray')

# %%


leftshifts = adf.getshifts(left)
rightshifts = adf.getshifts(right)
# %%
leftx = leftshifts[:, 1] + initialcenterleft[1]
rightx = rightshifts[:, 1] + initialcenterright[1]

positions = [leftx, rightx]

np.save('testlocs.npy', positions)
plt.plot(leftx)
plt.plot(rightx)
# %%
plt.plot(leftx-rightx)
# %%
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",'font.size': 9,
})
fields = ftf.findGuassVals(
    'run3.csv', 'run_MMStack_Pos0.ome.tif', 'tempguasssave')

BVals = fields[:, 2]


mperpix = 2.25e-6
fig, ax = plt.subplots(2, 1)

im = ax[0].imshow(ims[0], cmap='gray')
ax[0].set_ylim(900, 200)
ax[0].set_xlim(300, 1300)
line, = ax[0].plot([leftx[0], rightx[0]], [
                   initialcenterleft[0], initialcenterright[0]], 'ro')
scalebar = ScaleBar(mperpix, frameon=False, location='upper right', pad=0.5)
ax[0].add_artist(scalebar)
ax[0].axis('off')


delta = rightx-leftx

delta = delta-delta[0]

BLine, = ax[1].plot(BVals, delta, 'b.')
BPoint, = ax[1].plot([None, None], 'ro')
ax[1].set_xlim(0, 100)
ax[1].set_ylim(-5, 50)
ax[1].set_xlabel('B')
ax[1].set_ylabel(r'$d-d_o$')
plt.tight_layout()


def init():
    """
    This function gets passed to FuncAnimation.
    It initializes the plot axes
    """
    # Set plot limits etc

    # plt.tight_layout()
    return


def animate(i):
    im.set_data(ims[i])
    line.set_data([leftx[i], rightx[i]], [
                  initialcenterleft[0], initialcenterright[0]])
    BLine.set_data(BVals[:i], delta[:i])
    BPoint.set_data(BVals[i], delta[i])
    # line.set_data(irightline[1][0]+rightshifts[i,1],irightline[0])
    return line, im, BLine, BPoint,


ani = animation.FuncAnimation(
    fig, animate, frames=len(ims), interval=1, blit=True, repeat=False)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

plt.show()
# %%
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, extra_args=['-vcodec', 'libx264'])
ani.save('samplevid.mp4', writer=writer, dpi=200)
