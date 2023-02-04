# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 15:25:12 2022

@author: hamza
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
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

def peakfilter(y, prom = 0.1):
	
	#returns false values for where peaks and troughs are located
	
	normed = (np.max(y)-y)/np.max(y)
	peakdata = find_peaks(normed,prominence = prom,width=(None, 50))
	troughdata = find_peaks(-normed,prominence = prom,width=(None, 50))
	

	def excluder(y,peakarray):
		excludes = np.ones(len(y),dtype = bool)	
		if len(peakarray[0])==0:
			excludes = np.ones(len(y),dtype = bool)
		else:
			for i in range(len(peakarray[0])):
				centerloc = peakarray[0][i]
				width = peakarray[1]['widths'][i]
				llimit = int(centerloc-width)
				rlimit = int(centerloc+width)
				excludes[llimit:rlimit] = False
		return excludes
	
	pex = excluder(y,peakdata)
	tex = excluder(y,troughdata)
	excludearray = np.logical_and(pex,tex)
	return excludearray


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
	


tv = 7
#test= find_peaks(smoothspeeds[tv],prominence=1)[0]
plt.figure()
for tv in range(len(med_angles)):
	xs=np.arange(len(smoothspeeds[tv]))
	plt.plot(xs,smoothspeeds[tv]/med_angles[i],'-',label=tv)
	include = peakfilter(smoothspeeds[tv]/med_angles[i])
	plt.plot(xs[np.invert(include)],smoothspeeds[tv][np.invert(include)]/med_angles[i],'ko')
	#plt.axvline(test[0])
	if tv == int(len(med_angles)/2):
		plt.legend()
		plt.yscale('log')
		plt.figure()
plt.legend()
plt.yscale('log')

#%%

testfalses = peakfilter(smoothspeeds[9]/med_angles[i])
plt.plot(smoothspeeds[11]/med_angles[i],'.')
plt.plot(smoothspeeds[9][testfalses]/med_angles[i],'o')
plt.yscale('log')

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
		#print(popt)
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

def powerlaw(x,a):
	return  a*x**3

	
fig, ax = plt.subplots(figsize=(6, 5))
for i in sortedangles:
	if np.max(np.abs((dat[i][2][40:]-dat[i][2][20])*180/np.pi))<.4:
		ax.plot(dat[i][3][40:]*pixsize*1e6,smoothspeeds[i][40:]*1e6,'.',label = "{0:.1f}$^\circ$".format(med_angles[i]),color = pl.cm.inferno(scaledangles[i]))
		cleanx = dat[i][3][40:]*pixsize*1e6
		cleany = smoothspeeds[i][40:]*1e6
		popt,potx = curve_fit(powerlaw, cleanx,cleany ,p0=[.0003],bounds=[[0],[0.01]],maxfev=10000)
		#print(popt)
		#xsamples = np.linspace(0,700)
		
		#ax.plot(xsamples,powerlaw(xsamples,*popt),color = pl.cm.inferno(scaledangles[i]))
#ax.legend()
ax.set_xlabel(r'$d \ (\mathrm{\mu m})$')
ax.set_ylabel(r'$v (\mathrm{\mu m \ s^{-1}})$')
ax.set_xlim(100,1000)

ax.set_ylim(.1,10)

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

exclusions = np.loadtxt('manualkeep.csv',skiprows = 1,delimiter=',',dtype=int)

plt.figure(figsize=(5,4))
for i in sortedangles:
	#if np.max(np.abs((dat[i][2][40:]-dat[i][2][20])*180/np.pi))<.4:
	if (np.any(exclusions[:,0]==i)):
		ind = np.argwhere(exclusions[:,0]==i)
		val = int(exclusions[ind,1])
		if val>0:
			includes = peakfilter(smoothspeeds[i]*1e6/med_angles[i])
			includes[val:] = False
			x = dat[i][3][includes]*pixsize*1e6
			y = smoothspeeds[i][includes]*1e6/med_angles[i]
			#y = smoothspeeds[i][includes]*1e6
			plt.plot(x,y,'.',label = "{0:.1f}$^\circ$".format(med_angles[i]),color = pl.cm.inferno(scaledangles[i]))
	else:
		includes = peakfilter(smoothspeeds[i]*1e6/med_angles[i])
		x = dat[i][3][includes]*pixsize*1e6
		y = smoothspeeds[i][includes]*1e6/med_angles[i]
		#y = smoothspeeds[i][includes]*1e6
		plt.plot(x[20:],y[20:],'.',label = "{0:.1f}$^\circ$".format(med_angles[i]),color = pl.cm.inferno(scaledangles[i]))
		cleanx = dat[i][3][:]*pixsize*1e6
		cleany = smoothspeeds[i][:]*1e6/med_angles[i]
		#popt,potx = curve_fit(powerlaw, cleanx,cleany ,p0=[.0003,1.5,0],bounds=[[0,1,-100],[0.01,3.5,200]],maxfev=10000)
		#print(popt)
		xsamples = np.linspace(0,700)
		#plt.plot(xsamples,powerlaw(xsamples,*popt),color = pl.cm.inferno(scaledangles[i]))
plt.legend()
plt.xlabel(r'$d \ (\mathrm{\mu m})$')
plt.ylabel(r'$v/\theta (\mathrm{\mu m \ s^{-1}})$')
#plt.ylabel(r'$v (\mathrm{\mu m \ s^{-1}})$')
plt.yscale('log')
plt.xscale('log')
plt.savefig('vbtvd.png',dpi=900)
#plt.plot(x**2)

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
