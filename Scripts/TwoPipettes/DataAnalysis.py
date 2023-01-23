# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 15:25:12 2022

@author: hamza
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from scipy.signal import savgol_filter
import pynumdiff as pynd
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
#%%


from scipy.signal import butter, filtfilt
def quickfilter(x,y):
	params = np.polyfit(x, y,2)
	polytrend = np.polyval(params,x)
	b, a = butter(2,.1)
	smoothed = filtfilt(b,a,y-polytrend)
	smoothy = polytrend+smoothed
	return x, smoothy

def quickfilter2(x,y):
	params = np.polyfit(x, y,2)
	polytrend = np.polyval(params,x)
	b, a = butter(2,.1)
	smoothed = filtfilt(b,a,y-polytrend)
	smoothy = polytrend+smoothed
	return x, smoothy

#%%
runparams = np.loadtxt('runsparams.csv',skiprows=1,dtype=str,delimiter=',')
run_names = runparams[:,0]
run_time_steps = runparams[:,1].astype(float)



pixsize = 1.78e-6 #pixel size of camera in m
numRuns = len(run_names)
med_angles=np.zeros(numRuns)
dat=[None]*numRuns
speeds = [None]*numRuns
smoothspeeds = [None]*numRuns
for i in range(numRuns):
	leadtxt = run_names[i].split('.')[0]
	dat[i] = np.load(leadtxt+'.npy')
	med_angles[i] = 180/np.pi*np.median(dat[i][2])
	speeds[i] = np.abs(np.gradient(dat[i][0])/run_time_steps[i])*pixsize
	scratch, smoothspeeds[i] = quickfilter(dat[i][3],speeds[i])
	#[xlocs,pip_angles,sep_distances,d_to_centers]
#%%
plotorder = np.arange(11)
sortedangles = [x for _, x in sorted(zip(med_angles, plotorder))]
n = len(sortedangles)
scaledangles = np.array(med_angles)
scaledangles = (scaledangles-np.min(scaledangles))/np.max(scaledangles)
#%%
plt.figure()
for i in sortedangles:
	plt.plot(dat[i][3][20:-20]*pixsize*1e6,(dat[i][2][20:-20]-dat[i][2][20])*180/np.pi,label = "{0:.1f}$^\circ$".format(med_angles[i]),color = pl.cm.inferno(scaledangles[i]))
plt.legend()
plt.xlabel(r'$d \ (\mathrm{\mu m})$')
plt.ylabel(r'$\Delta\theta (\mathrm{^\circ})$')


#%%
n = 11
colors = pl.cm.inferno(np.linspace(0,1,n))

	
plt.figure(figsize=(5,4))
for i in sortedangles:
	if np.max(np.abs((dat[i][2][20:-20]-dat[i][2][20])*180/np.pi))<.4:
		plt.plot(dat[i][3][20:-20]*pixsize*1e6,smoothspeeds[i][20:-20]*1e6,label = "{0:.1f}$^\circ$".format(med_angles[i]),color = pl.cm.inferno(scaledangles[i]))
plt.legend()
plt.xlabel(r'$d \ (\mathrm{\mu m})$')
plt.ylabel(r'$v (\mathrm{\mu m \ s^{-1}})$')
plt.savefig('smoothexampledata.png',dpi=900)




