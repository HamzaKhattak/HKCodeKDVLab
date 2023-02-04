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
from scipy.optimize import curve_fit
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
#%%


from scipy.signal import butter, filtfilt
def quickfilter(x,y):
	params = np.polyfit(x, y,2)
	polytrend = np.polyval(params,x)
	b, a = butter(2,.2)
	smoothed = filtfilt(b,a,y-polytrend)
	smoothy = polytrend+smoothed
	return x, smoothy


def quickfilter2(x,y):
	params = np.polyfit(x, y,2)
	polytrend = np.polyval(params,x)
	b, a = butter(1,.05)
	smoothed = filtfilt(b,a,y-polytrend)
	smoothy = polytrend+smoothed
	return x, smoothy


def quickfilter3(x,y):
	return x, y

def peakfilter(x,y):
	'''
	what values to exclude from data due to troughs and peaks
	'''
	normed = (np.max(y)-y)/np.max(y)
	peakdata = find_peaks(normed,prominence = 0.1,width=(None, None))
	troughdata = find_peaks(-normed,prominence = 0.1,width=(None, None))
	if len test[0]==0:
		return excludes
	centerlocs = test[0]
	widths = test[1]['widths']
	plt.plot(peaktest)
	plt.axvline(centerlocs+widths)
	plt.axvline(centerlocs-widths)


runparams = np.loadtxt('runsparams.csv',skiprows=1,dtype=str,delimiter=',')
run_names = runparams[:,0]
run_time_steps = runparams[:,1].astype(float)



pixsize = 1.78e-6 #pixel size of camera in m
numRuns = len(run_names)
med_angles=np.zeros(numRuns)
dat=[None]*numRuns
speeds = [None]*numRuns
smoothspeeds = [None]*numRuns
leadtext = ['test']*numRuns
for i in range(numRuns):
	leadtxttemp = run_names[i].split('.')[0]
	leadtext[i] = leadtxttemp
	dat[i] = np.load(leadtxttemp+'.npy')
	med_angles[i] = 180/np.pi*np.median(dat[i][2])
	speeds[i] = np.abs(np.gradient(dat[i][0])/run_time_steps[i])*pixsize
	scratch, smoothspeeds[i] = quickfilter2(dat[i][3],speeds[i])
	#smoothspeeds[i] = speeds[i]
	#[xlocs,pip_angles,sep_distances,d_to_centers]
	
from scipy.signal import find_peaks

tv = 7
#test= find_peaks(smoothspeeds[tv],prominence=1)[0]
for tv in range(len(med_angles)):
	plt.plot(smoothspeeds[tv],'.',label=tv)
	#plt.axvline(test[0])
plt.legend()
plt.yscale('log')

#%%
from scipy.signal import find_peaks


#%%
plotorder = np.arange(len(med_angles))
sortedangles = [x for _, x in sorted(zip(med_angles, plotorder))]
n = len(sortedangles)
scaledangles = np.array(med_angles)
scaledangles = (scaledangles-np.min(scaledangles))/np.max(scaledangles)
#%%
plt.figure()
for i in sortedangles:
	plt.plot(dat[i][3][40:]*pixsize*1e6,(dat[i][2][40:]-dat[i][2][20])*180/np.pi,label = "{0:.1f}$^\circ$".format(med_angles[i]),color = pl.cm.inferno(scaledangles[i]))
plt.legend()
plt.xlabel(r'$d \ (\mathrm{\mu m})$')
plt.ylabel(r'$\Delta\theta (\mathrm{^\circ})$')
plt.savefig('angles.png',dpi=900)


#%%

n=11
colors = pl.cm.inferno(np.linspace(0,1,n))

def powerlaw(x,a,b):
	return  a*x**b

	
fig, ax = plt.subplots(figsize=(6, 5))
for i in sortedangles:
	if np.max(np.abs((dat[i][2][40:]-dat[i][2][20])*180/np.pi))<.4:
		ax.plot(dat[i][3][40:]*pixsize*1e6,smoothspeeds[i][40:]*1e6,'.',label = "{0:.1f}$^\circ$".format(med_angles[i]),color = pl.cm.inferno(scaledangles[i]))
		cleanx = dat[i][3][40:]*pixsize*1e6
		cleany = smoothspeeds[i][40:]*1e6
		#popt,potx = curve_fit(powerlaw2, cleanx,cleany ,p0=[.0003,1.5,0,0],bounds=[[0,1,-100,-100],[0.01,3.5,200,100]],maxfev=10000)
		#popt,potx = curve_fit(powerlaw, cleanx,cleany ,p0=[.0003,1.5,0],bounds=[[0,1,-100],[0.01,3.5,200]],maxfev=10000)
		print(popt)
		#xsamples = np.linspace(0,700)
		#ax.plot(xsamples,powerlaw(xsamples,*popt),color = pl.cm.inferno(scaledangles[i]))
#ax.legend()
ax.set_xlabel(r'$d \ (\mathrm{\mu m})$')
ax.set_ylabel(r'$v (\mathrm{\mu m \ s^{-1}})$')
ax.set_xlim(0,)

#plt.yscale('log')
#plt.xscale('log')
plt.savefig('vvd.png',dpi=900)

#%%
import matplotlib as mpl

n = len(med_angles)
anglediffs = [None]*n
for i in range(n):
	anglediffs[i] = np.max(np.abs((dat[i][2][40:]-dat[i][2][20])*180/np.pi))
colors = pl.cm.inferno(np.linspace(0,1,n))

def powerlaw(x,a,b,c):
	return  a*x**b+c

def powerlaw2(x,a,b,c,d):
	return  a*((x-d)*(x>d))**b+c

	
fig, ax = plt.subplots(figsize=(6, 5))
for i in sortedangles:
	if np.max(np.abs((dat[i][2][40:]-dat[i][2][20])*180/np.pi))<.4:
		ax.plot(dat[i][3][40:]*pixsize*1e6,smoothspeeds[i][40:]*1e6,'.',label = "{0:.1f}$^\circ$".format(med_angles[i]),color = pl.cm.inferno(scaledangles[i]))
		cleanx = dat[i][3][40:]*pixsize*1e6
		cleany = smoothspeeds[i][40:]*1e6
		#popt,potx = curve_fit(powerlaw2, cleanx,cleany ,p0=[.0003,1.5,0,0],bounds=[[0,1,-100,-100],[0.01,3.5,200,100]],maxfev=10000)
		#popt,potx = curve_fit(powerlaw, cleanx,cleany ,p0=[.0003,1.5,0],bounds=[[0,1,-100],[0.01,3.5,200]],maxfev=10000)
		#print(popt)
		#xsamples = np.linspace(0,700)
		#ax.plot(xsamples,powerlaw(xsamples,*popt),color = pl.cm.inferno(scaledangles[i]))
#ax.legend()
ax.set_xlabel(r'$d \ (\mathrm{\mu m})$')
ax.set_ylabel(r'$v (\mathrm{\mu m \ s^{-1}})$')
#ax.set_xlim(100,1000)

#ax.set_ylim(.1,10)

cmap = mpl.cm.inferno
norm = mpl.colors.Normalize(vmin=np.min(med_angles), vmax=np.max(med_angles))

cb1 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),orientation='horizontal', label=r'$\theta$')

plt.yscale('log')
plt.xscale('log')
plt.savefig('vvd.png',dpi=900)
#%%
n = len(med_angles)
anglediffs = [None]*n
for i in range(n):
	anglediffs[i] = np.max(np.abs((dat[i][2][40:]-dat[i][2][20])*180/np.pi))
colors = pl.cm.inferno(np.linspace(0,1,n))

exclusions = np.loadtxt('manualkeep.csv',skiprows = 1,delimiter=',',dtype=np.int)

plt.figure(figsize=(5,4))
for i in sortedangles:
	#if np.max(np.abs((dat[i][2][40:]-dat[i][2][20])*180/np.pi))<.4:
	if (np.any(exclusions[:,0]==i)):
		ind = np.argwhere(exclusions[:,0]==i)
		val = np.int(exclusions[ind,1])
		if val>0:
			plt.plot(dat[i][3][:val]*pixsize*1e6,smoothspeeds[i][:val]*1e6/med_angles[i],'.',label = "{0:.1f}$^\circ$".format(med_angles[i]),color = pl.cm.inferno(scaledangles[i]))
	else:
		plt.plot(dat[i][3][40:-40]*pixsize*1e6,smoothspeeds[i][40:-40]*1e6/med_angles[i],'.',label = "{0:.1f}$^\circ r{1}$".format(med_angles[i],i),color = pl.cm.inferno(scaledangles[i]))
		cleanx = dat[i][3][:]*pixsize*1e6
		cleany = smoothspeeds[i][:]*1e6/med_angles[i]
		#popt,potx = curve_fit(powerlaw, cleanx,cleany ,p0=[.0003,1.5,0],bounds=[[0,1,-100],[0.01,3.5,200]],maxfev=10000)
		#print(popt)
		xsamples = np.linspace(0,700)
		#plt.plot(xsamples,powerlaw(xsamples,*popt),color = pl.cm.inferno(scaledangles[i]))
plt.legend()
plt.xlabel(r'$d \ (\mathrm{\mu m})$')
plt.ylabel(r'$v/\theta (\mathrm{\mu m \ s^{-1}})$')
plt.xlim(100,1000)
plt.ylim(0.1,10)
plt.yscale('log')
plt.xscale('log')
plt.savefig('vbtvd.png',dpi=900)
#plt.plot(x**2)
#%%

	if 
#%%
n = 11
colors = pl.cm.inferno(np.linspace(0,1,n))



plt.figure(figsize=(5,4))
for i in sortedangles:
	if np.max(np.abs((dat[i][2][40:]-dat[i][2][20])*180/np.pi))<.4:
		plt.plot(dat[i][3][40:]*pixsize*1e6,smoothspeeds[i][40:]*1e6/dat[i][2][40:]/180*np.pi,label = "{0:.1f}$^\circ$".format(med_angles[i]),color = pl.cm.inferno(scaledangles[i]))
plt.legend()
plt.xlabel(r'$d \ (\mathrm{\mu m})$')
plt.ylabel(r'$v/\theta (\mathrm{\mu m \ s^{-1}})$')
plt.ylim(0,)
plt.xlim(0,)
#plt.yscale('log')
#plt.xscale('log')
plt.savefig('vbtvd.png',dpi=900)

#%%
n = 12
colors = pl.cm.inferno(np.linspace(0,1,n))



plt.figure(figsize=(5,4))
for i in sortedangles:
	if np.max(np.abs((dat[i][2][40:]-dat[i][2][20])*180/np.pi))<.5:
		vwithmean = smoothspeeds[i][40:]*1e6/med_angles[i]
		vwithpoint = smoothspeeds[i][40:]*1e6/dat[i][2][40:]/180*np.pi
		plt.plot(dat[i][3][40:]*pixsize*1e6, vwithmean-vwithpoint,'.',label = "{0:.1f}$^\circ$".format(med_angles[i]),color = pl.cm.inferno(scaledangles[i]))
plt.legend()
plt.ylim(-2,1)
plt.xlabel(r'$d \ (\mathrm{\mu m})$')
plt.ylabel(r'$v/\theta - v/\theta_o  (\mathrm{\mu m \ s^{-1}})$')
plt.savefig('vbtvd.png',dpi=900)
