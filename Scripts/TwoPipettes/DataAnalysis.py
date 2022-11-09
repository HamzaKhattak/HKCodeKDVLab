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
runparams = np.loadtxt('runsparams.csv',skiprows=1,dtype=str,delimiter=',')
run_names = runparams[:,0]
run_time_steps = runparams[:,1].astype(float)

pixsize = 2.25e-6 #pixel size of camera in m
numRuns = len(run_names)
med_angles=np.zeros(numRuns)
dat=[None]*numRuns
speeds = [None]*numRuns
for i in range(numRuns):
	leadtxt = run_names[i].split('.')[0]
	dat[i] = np.load(leadtxt+'.npy')
	med_angles[i] = 180/np.pi*np.median(dat[i][2])
	speeds[i] = np.abs(np.gradient(dat[i][0])/run_time_steps[i])*pixsize

#[xlocs,pip_angles,sep_distances,d_to_centers]

#%%
plotorder = np.arange(11)
sortedangles = [x for _, x in sorted(zip(med_angles, plotorder))]

#%%
plt.figure()
for i in sortedangles:
	plt.plot(dat[i][3][20:-20]*pixsize*1e6,(dat[i][2][20:-20]-dat[i][2][20])*180/np.pi,label = "{0:.1f}$^\circ$".format(med_angles[i]))
plt.legend()
plt.xlabel(r'$d \ (\mathrm{\mu m})$')
plt.ylabel(r'$\Delta\theta (\mathrm{^\circ})$')


#%%
plt.figure()
for i in sortedangles:
	plt.plot(dat[i][3][20:-20]*pixsize*1e6,speeds[i][20:-20]*1e6,label = "{0:.1f}$^\circ$".format(med_angles[i]))
plt.legend()
plt.xlabel(r'$d \ (\mathrm{\mu m})$')
plt.ylabel(r'$v (\mathrm{\mu m \ s^{-1}})$')


