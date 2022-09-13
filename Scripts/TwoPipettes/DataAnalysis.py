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
for i in range(len(forward)):
	if forward[i]==False:
		directionlist[i]='Backward'
	else:
		directionlist[i] = 'Forward'

#%%

alldats = [ex1dat,ex2dat,ex3dat,ex4dat,ex5dat,ex6dat,ex7dat]
#sort the list based on mean angle for ease of plotting
meanangles = [np.mean(i[1]) for i in alldats]


sorteddats = [x for _, x in sorted(zip(meanangles, alldats))]
sortedlabels = [x for _, x in sorted(zip(meanangles, directionlist))]

for i in range(len(sorteddats)):
	med_angle = np.median(sorteddats[i][1])
	angle_sdev = np.std(sorteddats[i][1])
	smoothed = savgol_filter(sorteddats[i][-1][20:-20],31,3)
	derives = np.gradient(sorteddats[i][0][20:-20])
	plt.plot(smoothed[50:-20],derives[50:-20],'.',label =rf'{med_angle*180/np.pi:.1f} $\pm$ {angle_sdev*180/np.pi:.1f}$\degree$ {sortedlabels[i]:s}')
plt.legend()
plt.xlabel('distance from center')
plt.ylabel('speed')
#%%

#%%
xt, xdt = pynd.iterative_velocity(ex3dat[-1], 1e-3, [100,.2])
xdt=xdt/1e3
plt.plot(ex3dat[-1],'.')
plt.plot(xt)
#%%
