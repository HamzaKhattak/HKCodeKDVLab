# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 15:25:12 2022

@author: hamza
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pynumdiff as pynd
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
#%%
pixsize = 2.25e-6 #pixel size of camera in m
runparams = np.loadtxt('RunInfo.csv',delimiter=',',skiprows=1)
secperframes = runparams[:,1]
directions = runparams[:,2]
med_angles=np.zeros(len(directions))
dat=[None]*11
speeds = [None]*11
for i in range(11):
	dat[i] = np.load('drop_'+str(i+1)+'.npy')
	med_angles[i] = 180/np.pi*np.median(dat[i][1])*directions[i]
	speeds[i] = np.abs(np.gradient(dat[i][0])/secperframes[i])*pixsize

#[xlocs,pip_angles,sep_distances,d_to_centers]

#%%
plotorder = np.arange(11)
sortedangles = [x for _, x in sorted(zip(med_angles, plotorder))]

#%%
for i in sortedangles:
	plt.plot(dat[i][2][20:-20]*pixsize*1e6,(dat[i][1][20:-20]-dat[i][1][20])*180/np.pi,label = "{0:.1f}$^\circ$".format(med_angles[i]))
plt.legend()
plt.xlabel(r'$d \ (\mathrm{\mu m})$')
plt.ylabel(r'$\Delta\theta (\mathrm{^\circ})$')
#%%
for i in sortedangles:
	plt.plot(dat[i][2][20:-20]*pixsize*1e6,speeds[i][20:-20]*1e6,label = "{0:.1f}$^\circ$".format(med_angles[i]))
plt.legend()
plt.xlabel(r'$d \ (\mathrm{\mu m})$')
plt.ylabel(r'$v (\mathrm{\mu m \ s^{-1}})$')


