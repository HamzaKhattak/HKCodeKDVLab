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

ex1dat = np.load('exp1.npy')
ex2dat = np.load('exp2.npy')
ex3dat = np.load('exp3.npy')
ex4dat = np.load('exp4.npy')
ex5dat = np.load('exp6.npy')
ex6dat = np.load('exp7.npy')
ex7dat = np.load('exp8.npy')
#%%
forward =[True,False,True,True,True,False,True]
directionlist = [None]*len(forward)

manual_angles = np.array([1.7,.8,.81,3.5,1.1,1.4,1])

for i in range(len(forward)):
	if forward[i]==False:
		directionlist[i]='B'
	else:
		directionlist[i] = 'F'

#%%

alldats = [ex1dat,ex2dat,ex3dat,ex4dat,ex5dat,ex6dat,ex7dat]
#sort the list based on mean angle for ease of plotting
meanangles = [np.mean(i[1]) for i in alldats]


sorteddats = [x for _, x in sorted(zip(manual_angles, alldats))]
sortedlabels = [x for _, x in sorted(zip(manual_angles, directionlist))]
sortangles = np.sort(manual_angles)
fig = plt.figure(figsize=(6,4))
for i in range(len(sorteddats)):
	med_angle = np.median(sorteddats[i][1])
	med_angle = sortangles[i]
	angle_sdev = np.std(sorteddats[i][1])
	smoothed = savgol_filter(sorteddats[i][2][20:-20],31,3)
	derives = np.abs(np.gradient(sorteddats[i][0][20:-20]))
	plt.plot(smoothed[50:-20]*2,derives[50:-20]*2,'.',label = "{0:.1f}$^\circ$ {1}".format(med_angle,sortedlabels[i]))
plt.legend()
plt.xlabel(r'$d \ (\mathrm{\mu m})$')
plt.ylabel(r'$v (\mathrm{\mu m \ s^{-1}})$')
plt.tight_layout()
plt.savefig('multiangleplot.png',dpi=900)
#%%

#%%
xt, xdt = pynd.iterative_velocity(ex3dat[-1], 1e-3, [100,.2])
xdt=xdt/1e3
plt.plot(ex3dat[-1],'.')
plt.plot(xt)
#%%
