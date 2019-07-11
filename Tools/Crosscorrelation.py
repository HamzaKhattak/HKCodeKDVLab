import numpy as np
from scipy import signal
'''
takes in two arrays and finds the cross correlation array of shifts
'''
a1=[0,0.5,1,0.5,0,0,0]
a2=[0,0,0,0.5,1,0.5,0]
a3=[0,0.5,1,0.5,0,0,0]

a4=[0,1,0]
a5=[0,0,1]

xrang=np.linspace(-10,10,num=100)

g1=np.ga

def crosscorrelator(veca,vecb):
	a = (a - np.mean(a)) / (np.std(a) * len(a))
	b = (b - np.mean(b)) / (np.std(b))
	c = np.correlate(a, b, 'full')

def centerfinder(a):
	
#%%
cresult=np.correlate(a1,a3)
print(cresult)
spresult=signal.correlate(a4,a5,"full")  
print(spresult)  