# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 13:54:27 2024

@author: hamza
"""

import numpy as np
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series  # for convenience

import pims
import trackpy as tp

from sklearn.neighbors import KDTree

#%%
def savelistnp(filepath,data):
	'''
	Saves lists of numpy arrays using pickle so they don't become objects
	'''
	with open(filepath, 'wb') as outfile:
		   pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)

def openlistnp(filepath):
	'''
	Opens lists of numpy arrays using pickle
	'''
	with open(filepath, 'rb') as infile:
	    result = pickle.load(infile)
	return result

#%%

locations = openlistnp('initialrunpositions.pik')
#%%



#%%
'''
Code that finds nearest neighbours without doing any tracking, smoothing etc. 
'''
testlocs = locations[400]

def findNN(points,rad):
	'''
	This code uses a KD tree to quickly find how many points within
	a given distance of each point neigbour points. Should use rad>droplet radius
	to account for any errors in finding the position
	returns the indices of the nearest neighbours as well as the number of nearest neighbours
	'''
	tree = KDTree(points, leaf_size=2)
	nn_indices = tree.query_radius(points, r=rad) #Find KD tree which also gives point indices
	numNN = np.array([len(i)-1 for i in nn_indices]) #Find the number of nearest neighbours per site
	#find fractions of nearest neighbours with given number of neighbours
	fractionNN = [None]*7
	for j in np.arange(0,7,1):
		fractionNN[j] = np.count_nonzero(numNN==j)/len(numNN)
	return nn_indices, numNN, fractionNN

def findNNforsequence(seqofpoints,rad):
	'''
	returns the indices of the nearest neighbours as well as the number of nearest neighbours


	'''
	all_nn_indices = [None]*len(seqofpoints) # gives actual location
	all_nns = [None]*len(seqofpoints)
	fractionNN = np.zeros((len(seqofpoints),7))
	for i in range(len(seqofpoints)):
		all_nn_indices[i] , all_nns[i],fractionNN[i] = findNN(seqofpoints[i],15) 

	return all_nn_indices,all_nns,fractionNN

#%%

test,test,fractions = findNNforsequence(locations, 15)

for i in np.arange(0,7,1):
	plt.plot(fractions[:,i],label=i)
plt.legend(title='Nearest neighbours')
plt.xlabel('time (arb, can convert to G later)')
plt.ylabel('fraction')

#%%

'''
This is the section of code to track individual droplets if needed
'''