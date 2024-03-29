# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 15:25:12 2022

@author: hamza
"""
import pickle, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pl
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import pynumdiff as pynd
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})
#%%



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

def peakfilter(y, prom = 2):
	
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
#run_names = [i[:-1] for i in run_names]

run_time_steps = runparams[:,1].astype(float)



pixsize = 2.24e-6 #pixel size of camera in m
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
	


#test= find_peaks(smoothspeeds[tv],prominence=1)[0]
plt.figure()
for tv in range(len(med_angles)):
	xs=np.arange(len(smoothspeeds[tv]))
	include = peakfilter(smoothspeeds[tv]/med_angles[i])
	plt.plot(xs[np.invert(include)],smoothspeeds[tv][np.invert(include)]/med_angles[i],'ko')
	plt.plot(xs,smoothspeeds[tv]/med_angles[i],'-',label=tv,linewidth=3)
	#plt.axvline(test[0])
	if tv == int(len(med_angles)/2):
		plt.legend()
		#plt.yscale('log')
		#plt.ylim(.00001,)
		plt.figure()
plt.legend()
#plt.yscale('log')
#plt.ylim(.00001,)
#%%
plt.figure()
for tv in range(len(med_angles)):
	xs=dat[tv][3]
	include = peakfilter(smoothspeeds[tv]/med_angles[i])
	plt.plot(xs[np.invert(include)],smoothspeeds[tv][np.invert(include)]/med_angles[i],'ko')
	plt.plot(xs,smoothspeeds[tv]/med_angles[i],'-',label=tv)
	#plt.axvline(test[0])
	if tv == int(len(med_angles)/2):
		plt.legend()
		plt.yscale('log')
		plt.figure()
plt.legend()
plt.yscale('log')

#%%
cleanedx = [None]*len(med_angles)
cleanedy = [None]*len(med_angles)
exclusions = np.loadtxt('manualkeep.csv',skiprows = 1,delimiter=',',dtype=int)

for i in range(len(med_angles)):
	#if np.max(np.abs((dat[i][2][40:]-dat[i][2][20])*180/np.pi))<.4:
	includes = peakfilter(smoothspeeds[i]*1e6)
	if (np.any(exclusions[:,0]==i)):
		ind = np.argwhere(exclusions[:,0]==i)
		limL = int(exclusions[ind,1])
		limR = int(exclusions[ind,2])
		if limR == 0:
			x = np.mean(dat[i][3][includes]*pixsize*1e6)
			y = np.mean(smoothspeeds[i])
		elif limR>0:
			includes = peakfilter(smoothspeeds[i]*1e6)
			includes[limR:] = False
			if limL>0:
				includes[:limL] = False
			x = dat[i][3][includes]*pixsize*1e6
			y = smoothspeeds[i][includes]*1e6
	else:
		includes = peakfilter(smoothspeeds[i]*1e6)
		x = dat[i][3][includes]*pixsize*1e6
		y = smoothspeeds[i][includes]*1e6
	cleanedx[i] = x
	cleanedy[i] = y


def savelistnp(filepath,data):
	'''
	Saves lists of numpy arrays using pickle so they don't become objects
	'''
	with open(filepath, 'wb') as outfile:
		pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)

#datsavename=os.path.basename(os.path.dirname(os.getcwd()))+'dat'
datsavename=os.path.basename(os.getcwd())+'dat'
savelistnp(datsavename,[cleanedx,cleanedy,med_angles])

#%%
plotorder = np.arange(len(med_angles))
sortedangles = [x for _, x in sorted(zip(med_angles, plotorder))]
n = len(sortedangles)
scaledangles = np.array(med_angles)
scaledangles = (scaledangles-np.min(scaledangles))/np.max(scaledangles)
plt.figure() 
for i in sortedangles:
	plt.plot(dat[i][3][40:]*pixsize*1e6,(dat[i][2][40:]-dat[i][2][20])*180/np.pi,label = "{0:.1f}$^\circ$".format(med_angles[i]),color = pl.cm.inferno(scaledangles[i]))
plt.legend()
plt.xlabel(r'$d \ (\mathrm{\mu m})$')
plt.ylabel(r'$\Delta\theta (\mathrm{^\circ})$')
plt.savefig('angles.png',dpi=900)


#%%

log = False
fits = False
byangle = True
transparent = True
onlyselect = False
whichtoplot = [13,10,0] #[7,1,12] #[13,10,0]
savename = 'im6.png'

def powerlaw(x,a):
	return  a*x**3

		


n = len(med_angles)
#colours = pl.cm.viridis(np.linspace(0,1,n))
colours = pl.cm.viridis(scaledangles)

allfits = np.array([])
xsamples = np.linspace(0,1000)
fig, ax = plt.subplots(figsize=(6, 5))
for i in sortedangles:
	x = cleanedx[i]
	y = cleanedy[i]
	if byangle:
		y=y/(med_angles[i]*np.pi/180)
	if onlyselect == False:
		#y = smoothspeeds[i][includes]*1e6
		if not (isinstance(x, float)):			
			if transparent ==False:
				ax.plot(x,y,'.',label = "{0:.1f}$^\circ {1:.1f}$".format(med_angles[i],i),color = colours[i])
			
			else:
				ax.plot([],'.',label = "{0:.1f}$^\circ$ {1:.1f}".format(med_angles[i],i),color = colours[i])
				ax.plot(x,y,'-',color = colours[i],linewidth=3.5,alpha=0.7)
			popt,potx = curve_fit(powerlaw, x,y ,p0=[.0003],bounds=[[0],[0.01]],maxfev=10000)
			allfits = np.append(allfits,popt[0])			
		if fits:
			#print(popt)
			ax.plot(xsamples,powerlaw(xsamples,*popt),'--',color = colours[i])
	if onlyselect == True:
		if any(whichtoplot==i):
			ax.plot(x,y,'.',label = "{0:.1f}$^\circ$".format(med_angles[i]),color = colours[i])
			popt,potx = curve_fit(powerlaw, x,y ,p0=[.0003],bounds=[[0],[0.01]],maxfev=10000)
			#allfits = allfits.append(popt)
			if fits:

				#print(popt)
				ax.plot(xsamples,powerlaw(xsamples,*popt),'--',color = colours[i])

sortedangles


if onlyselect==False:
	fitdatas = sortedangles
else:
	fitdatas = whichtoplot

allx=np.array([])
ally=np.array([])
for i in range(len(fitdatas)):
	y = cleanedy[i]
	if byangle ==True:
		y=y/(med_angles[i]*np.pi/180)
	allx = np.append(allx,cleanedx[i])
	ally = np.append(ally,y)

totalpopt,totalpotx = curve_fit(powerlaw, allx,ally ,p0=[.0003],bounds=[[0],[0.01]],maxfev=10000)

#ax.plot(xsamples,powerlaw(xsamples,np.mean(allfits)),'--',color = 'k')
ax.plot(xsamples,powerlaw(xsamples,*totalpopt),'--',color = 'k')



ax.set_ylim(0,)
ax.set_xlim(0,)
ax.legend(loc = 'upper left')
ax.set_xlabel(r'$d \ (\mathrm{\mu m})$')

if byangle:
	ax.set_ylabel(r'$v/\theta (\mathrm{\mu m \ s^{-1}})$')
else:
	plt.ylabel(r'$v (\mathrm{\mu m \ s^{-1}})$')


if log:
	ax.set_yscale('log')
	ax.set_xscale('log')
plt.savefig(savename,dpi=900)

cmap = mpl.cm.inferno
norm = mpl.colors.Normalize(vmin=np.min(med_angles), vmax=np.max(med_angles))

#cb1 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),orientation='horizontal', label=r'$\theta$')