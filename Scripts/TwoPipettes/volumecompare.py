# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 19:30:06 2023

@author: WORKSTATION
"""

import pickle
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


def openlistnp(filepath):
	'''
	Opens lists of numpy arrays using pickle
	'''
	with open(filepath, 'rb') as infile:
	    result = pickle.load(infile)
	return result


dats=[vol1,vol2,vol3]
vols = [75, 146, 61]
labels = ['75 pL', '146 pL', '61 pL']
colours = ['b','m', 'g']

for i in range(len(labels)):
	print(i)
	for j in  range(len(dats[i][0])):
		d = dats[i][0][j]
		V = vols[i]
		speeds = dats[i][1][j]
		angles = dats[i][2][j]
		x = d/V**(1/3)
		y = speeds/angles
		if j==0:
			plt.plot(x,y,'.',color = colours[i], label=labels[i])
		else:
			plt.plot(x,y,'.',color = colours[i])


plt.ylabel(r'$v/\theta$')
plt.xlabel(r'$d/V^{1/3}$')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.ylim(.1,)
