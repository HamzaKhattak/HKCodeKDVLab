import numpy as np
from sklearn.neighbors import KDTree

#%%

'''
Code that finds nearest neighbours without doing any tracking, smoothing etc. 
'''


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

def findstrings(seqofpoints,dropletrad):
	'''
	Finds NN and bins to the catagories
	'''
	s,s,fractions = findNNforsequence(seqofpoints, dropletrad+3)
	disperse = fractions[:,0]
	packed = np.sum(fractions[:,-2:],axis=1)
	string = 1-packed-disperse
	return np.transpose([packed,string,disperse])