# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 19:30:06 2023

@author: WORKSTATION
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pl
from scipy.optimize import curve_fit
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})


def openlistnp(filepath):
	'''
	Opens lists of numpy arrays using pickle
	'''
	with open(filepath, 'rb') as infile:
	    result = pickle.load(infile)
	return result

vol1 = openlistnp('volume1dat')
vol2 = openlistnp('volume2dat')
vol3 = openlistnp('volume3dat')
vol4 = openlistnp('volume4dat')
vol5 = openlistnp('volume5dat')
def openlistnp(filepath):
	'''
	Opens lists of numpy arrays using pickle
	'''
	with open(filepath, 'rb') as infile:
	    result = pickle.load(infile)
	return result


dats=[vol1,vol2,vol3,vol4,vol5]  
vols = np.loadtxt('volumes.csv',skiprows = 1,delimiter=',',dtype=int)
vols=vols[:,1]

sphereR = (vols/10e12)**(1/3)*1e6
labels = [str(i) +'pL' for i in vols]

n = len(vols)
plotorder = np.arange(n)
colours = pl.cm.plasma_r(np.linspace(0,1,n))
sortedcolours = [x for _, x in sorted(zip(vols, colours))]
sortedplotorder = [x for _, x in sorted(zip(vols, plotorder))]

#Section for colours
#colours = ['r','m', 'g','b','c']
colours = sortedcolours

logvols = np.log(vols)
scaledvolumes = (vols-np.min(vols))/np.max(vols-np.min(vols))
scaledvolumes = (logvols-np.min(logvols)+.5)/np.max(logvols-np.min(logvols)+.5)
colours = pl.cm.plasma_r(scaledvolumes)


xsamples=np.linspace(0, 2000,num=10000)
def powerlaw(x,a):
	return  a*x**3
fig, ax = plt.subplots(figsize=(6, 5))
allpopt = np.array([])

byVolume = True
savename='im4.png'
x1 = 0
x2 = 200
#for i in [0,4]: #[3,1,0,2,4] [0,2]
for i in sortedplotorder:
	print(i)
	allx = np.array([])
	ally = np.array([])
	ax.plot([],'.',color = colours[i], label=labels[i])
	for j in  range(len(dats[i][0])):
		d = dats[i][0][j]
		V = vols[i]
		speeds = dats[i][1][j]
		angles = dats[i][2][j]
		
		if byVolume==True:	
			x = d/V**(1/3)
		else:
			x = d
		
		y = speeds/(angles*np.pi/180)
		
		x=x[y>2]
		y=y[y>2]
		allx = np.append(allx,x)
		ally = np.append(ally,y)
		
		ax.plot(x,y,'-',color = colours[i],alpha=0.5,markerfacecolor="None",markersize=5,linewidth=3)
	popt,potx = curve_fit(powerlaw, allx,ally ,p0=[.003],bounds=[[0],[0.1]],maxfev=10000)
	allpopt = np.append(allpopt, popt)
	#plt.axvline(2*sphereR[i],color=colours[i])
	#plt.plot(xsamples,powerlaw(xsamples,*popt),'--',color=colours[i])

ax.plot(xsamples,powerlaw(xsamples,np.mean(allpopt[allpopt!=np.min(allpopt)])),'k--')
ax.set_ylabel(r'$v/\theta \  \mathrm{(\mu m s^{-1}})$')

if byVolume ==True:
	ax.set_xlabel(r'$d/\Omega^{1/3} \ \mathrm{(\mu m/pL})$')

else:
	ax.set_xlabel(r'$d \ \mathrm{(\mu m})$')

ax.legend(loc='upper left')

#ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_ylim(0,500)
ax.set_xlim(x1,x2)

#ax.set_ylim(5,1000)
#ax.set_xlim(30,200)
#plt.tight_layout()
plt.savefig(savename,dpi=900)
