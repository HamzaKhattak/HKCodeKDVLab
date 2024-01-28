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
from win11toast import notify
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
#Import the data
locations = openlistnp('initialrunpositions.pik')

gaussvals = np.loadtxt('FrametoGauss.csv',delimiter=',')[:,2]


#%%
'''
Code that finds nearest neighbours without doing any tracking, smoothing etc. 
'''
testlocs = locations[10]

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
		all_nn_indices[i] , all_nns[i],fractionNN[i] = findNN(seqofpoints[i],rad) 

	return all_nn_indices,all_nns,fractionNN

#%%
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",'font.size': 9,
})
test,test,fractions = findNNforsequence(locations, 52)
finalframe = 200
for i in np.arange(0,7,1):
	plt.plot(gaussvals[:finalframe],fractions[:,i][:finalframe],label=i)
plt.xlim(0,40)
plt.legend(title='Nearest neighbours')
plt.xlabel(r'$B$ (G)')
plt.ylabel('fraction')

#%%

'''
This is the section of code to track individual droplets if needed
'''
def converttoDataFrame(inputlocationarray):
	'''
	Trackpy likes the input to be a Pandas dataframe ()
	'''
	frame = np.array([],dtype=float)
	x= np.array([],dtype=float)
	y=np.array([],dtype=float)
	#convert to dataframe:
	for i in range(len(inputlocationarray)):
		frame = np.append(frame, i*np.ones(len(inputlocationarray[i])))
		x = np.append(x,inputlocationarray[i][:,0])
		y = np.append(y,inputlocationarray[i][:,1])
	
	
	pddat = pd.DataFrame({'frame': frame, 'x': x, 'y': y})
	return pddat

posdataframe = converttoDataFrame(locations)

#%%
'''
#This section of code is for trackpy if it is needed
t1 = tp.link(posdataframe, 10,memory=50)
#%%

t2 = tp.filter_stubs(t1,50)
print('Before:', t1['particle'].nunique())
print('After:', t2['particle'].nunique())
d = tp.compute_drift(t2)

tm = tp.subtract_drift(t2.copy(), d)
tp.plot_traj(t2)
ax = tp.plot_traj(tm)
plt.show()
#%%
t4 = t2.loc[t2['particle'] == 15]
print(len(t4))
plt.plot(t4.x,'.')
#%%

plt.imshow(correctedims[0],cmap='gray')

ind = [None]*460

for i in np.arange(0,460,1):
	ind[i] = t2.loc[t2['particle'] == i]
	print(i,':',len(ind))
	if len(ind[i])<700:
		print('test')
		plt.plot(ind[i].y,ind[i].x)


#%%
mask = np.logical_and(my_array[:, 1] >= 55, my_array[:, 1] <= 65)
testfilt = t2.iloc[np.where(np.count_nonzero(t2['particle'] == np.arange())<700)]

#%%


#%%
indices= np.where(len(t2[['frame','x','y']].iloc[np.where(t2['particle']==i)]))

#%%
test = t2[['x', 'y','particle']].iloc[np.where(t2['frame']==0)]


test2 = t2[['frame','x','y']].iloc[np.where(t2['particle']==100)]

plt.plot(np.array(test2.x))

plt.plot(test2.frame,test2.x)
print(len(test2))

#%%
import nest_asyncio
import asyncio

from threading import Thread

notify('Done')


#%%
for i in range(t2['particle'].nunique()):
	print(i,':',len(t2[['frame','x','y']].iloc[np.where(t2['particle']==i)]))
	

'''
