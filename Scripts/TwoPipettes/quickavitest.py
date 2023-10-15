# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 13:19:39 2023

@author: WORKSTATION
"""
import numpy as np
import matplotlib.pyplot as plt
import pims
v = pims.PyAVReaderIndexed('run1.avi')
#plt.imshow(v[-1,:,:,0])  # a 2D numpy array representing the last frame
plt.imshow(v[0][:,:,0],cmap='gray')
plt.imshow(v[-1][:,:,0],cmap='gray',alpha=0.5)
