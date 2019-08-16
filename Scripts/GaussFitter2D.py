# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 16:10:29 2019

@author: WORKSTATION
"""
import os, sys, importlib
import scipy.optimize as opt
import numpy as np
import pylab as plt2
import matplotlib.pyplot as plt
import imageio
from mpl_toolkits.mplot3d import Axes3D
#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"E:\SpeedScan"


os.chdir(CodeDR) #Set  current working direcotry to the code directory


sys.path.append('./Tools') #Add the tools to the system path so modules can be imported

#Import required modules
import DropletprofileFitter as df
importlib.reload(df)
import Crosscorrelation as crco
importlib.reload(crco)
import ImportTools as ito 
importlib.reload(ito)
import EdgeDetection as ede
importlib.reload(ede)

#Remove to avoid cluttering path
sys.path.remove('./Tools') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)
#%%
image=imageio.imread('data.tif')
imagecropped=image[820:840,400:420]
plt2.imshow(imagecropped)
plt2.colorbar()
#%%

#define model function and pass independant variables x and y as a list
def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata_tuple                                                        
    xo = float(xo)                                                              
    yo = float(yo)                                                              
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)   
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)    
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)   
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)         
                        + c*((y-yo)**2)))                                   
    return g.ravel()

# Create x and y indices
x = np.linspace(0, 19, 20)
y = np.linspace(0, 19, 20)
X = np.meshgrid(x, y)
'''
#create data
data = twoD_Gaussian((x, y), 3, 100, 100, 20, 40, 0, 10)

# plot twoD_Gaussian data generated above
plt.figure()
plt.imshow(data.reshape(20, 20))
plt.colorbar()
'''
# add some noise to the data and try to fit the data generated beforehand
initial_guess = (.1,10,10,2,2,0,.05)

#data_noisy = data + 0.2*np.random.normal(size=data.shape)
data_noisy = np.ravel(imagecropped)
popt, pcov = opt.curve_fit(twoD_Gaussian,X, data_noisy, p0=initial_guess)

data_fitted = twoD_Gaussian(X, *popt)

fig, ax = plt.subplots(1, 1)
#ax.hold(True)
im=ax.imshow(data_noisy.reshape(20, 20), cmap=plt.cm.jet, origin='bottom',
    extent=(x.min(), x.max(), y.min(), y.max()))
ax.contour(x, y, data_fitted.reshape(20, 20), 8, colors='w')
cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.show()

#%%

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(X[0], X[1], imagecropped)
ax.plot_wireframe(X[0], X[1], data_fitted.reshape(20, 20),color='r')
