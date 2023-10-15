# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 21:02:49 2023

@author: hamza
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import imageio
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
import imageio as io

from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
from matplotlib_scalebar.scalebar import ScaleBar

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
testim=io.imread(filename)

cropyes = True

croplx = 10
croply = 287
xrange = 1100
yrange = 600

'''
croplx = 0
croply = 200
xrange = len(testim[0])
yrange = 800
'''
if cropyes == True:
	testim = testim[croply:croply+yrange,croplx:croplx+xrange]

mperpix = 1.78e-6  #the meters per pixel for the top angle

fig, ax = plt.subplots(1,1,figsize=(2.25,1.125))
fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
ax.axis('off')
ax.margins(0,0)
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())
mainim = ax.imshow(testim,cmap=plt.cm.binary_r,aspect='equal')
#mainim.set_clim(0, 256)
#scalebar = ScaleBar(mperpix,frameon=False,location='lower right',label_formatter = lambda x, y:'',border_pad=1,scale_loc='top')  
scalebar = ScaleBar(mperpix,frameon=False,location='upper right',pad=0.5) 
ax.add_artist(scalebar)

savename = os.path.split(os.path.splitext(filename)[0])[1]

plt.savefig(savename+'mod.pdf', dpi=900,pad_inches = 0, bbox_inches='tight')