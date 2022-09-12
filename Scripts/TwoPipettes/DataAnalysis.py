# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 15:25:12 2022

@author: hamza
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pynumdiff as pynd
#%%
plt.plot(pip_angles*180/np.pi)
plt.xlabel('time')
plt.ylabel('pipette_angle')
#%%
plt.plot(d_to_centers)
smoothed = savgol_filter(d_to_centers,31,3)
plt.xlabel('time')
plt.ylabel('distance from pipette cross point')


def smoothconvolve(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

csmooth = smoothconvolve(d_to_centers,10)[10:-10]
plt.plot(csmooth)

#%%
plt.plot(xlocs,np.abs(np.gradient(xlocs)))
plt.xlabel('position')
plt.ylabel('speeed')
#%%
xt, xdt = pynd.iterative_velocity(d_to_centers, 1e-3, [100,.2])
xdt=xdt/1e3
plt.plot(d_to_centers,'.')
plt.plot(xt)
#%%
plt.plot(smoothed,np.abs(np.gradient(smoothed)),label='savgol')
plt.plot(d_to_centers,np.abs(np.gradient(d_to_centers)),'.',label='no smoothing')
plt.plot(xt,np.abs(xdt),label ='regularization')
plt.plot(csmooth,np.abs(np.gradient(csmooth)),label ='convolve')
plt.xlabel('distance from pipette cross point')
plt.ylabel('speed')
plt.legend()

#%%
mean_angle = np.median(pip_angles)
div_mean = np.std(pip_angles)
uline= np.poly1d(upper_line_params[0])
lline= np.poly1d(lower_line_params[0])

sep_d2 = np.abs(uline(xlocs)-lline(xlocs))
dc1 = sep_d2/(2*np.tan(mean_angle/2))


smoothdist = savgol_filter(sep_distances,31,3)
smoothangles = savgol_filter(pip_angles,11,2)
dc2 = smoothdist/(2*np.tan(smoothangles/2))


plt.plot(pip_angles*180/np.pi)
plt.plot(smoothangles*180/np.pi)

#%%
cond=np.abs(pip_angles-mean_angle)<.5*div_mean

plt.plot(dc1,np.gradient(dc1),'r.',label='no smooth, single line')
plt.plot(dc2[cond],np.abs(np.gradient(dc2))[cond],'b.',label='smooth, line per frame')
plt.legend()
