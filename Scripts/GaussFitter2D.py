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

#Read and crop image as needed
image=imageio.imread('data.tif')
imagecropped=image[825:837,407:417]
'''
#If synthetic data needs to be created
x=np.linspace(0, 49,50)
y=np.linspace(0, 49,50)
X = np.meshgrid(x, y)
imagecropped = twoD_Gaussian(X,*[1,25,25,5,10,.5,.05]).reshape(50, 50)
imagecropped=imagecropped++ 0.02*np.random.normal(size=imagecropped.shape)
plt2.imshow(imagecropped)
plt2.colorbar()
'''
#%%
#Method one with ravel
'''
This method turns the 2D array into one long 1D array for fitting
'''
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

# Create x and y indices and the required meshgrid
xlen=imagecropped.shape[1]
ylen=imagecropped.shape[0]
x=np.linspace(0, xlen-1,xlen)
y=np.linspace(0, ylen-1,ylen)
X = np.meshgrid(x, y)

# Provide an initial guess based on the cropped image, could use max to make this better
initial_guess = (1,25,25,2,2,0,.05)

#data_noisy = data + 0.2*np.random.normal(size=data.shape)
data_noisy = np.ravel(imagecropped)
popt, pcov = opt.curve_fit(twoD_Gaussian,X, data_noisy, p0=initial_guess,maxfev = 10000)

data_fitted = twoD_Gaussian(X, *popt)

fig, ax = plt.subplots(1, 1)
#ax.hold(True)
im=ax.imshow(data_noisy.reshape(ylen, xlen), cmap=plt.cm.jet, origin='bottom',
    extent=(x.min(), x.max(), y.min(), y.max()))
ax.contour(x, y, data_fitted.reshape(ylen, xlen), 8, colors='w')
cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.show()

#%%
'''
This section does a 2D fit using the curve fit 2D fit rather than raveling
'''
#Create x and y in the proper shapes ie [[x1,y1,z1],[x2,y2,z2],....]
x=np.linspace(0, xlen-1,xlen)
y=np.linspace(0, ylen-1,ylen)

xyzarr=np.zeros((x.size*y.size,3))

xyzarr[:,0]=np.repeat(x,y.size)
xyzarr[:,1]=np.tile(y,x.size)
xyzarr[:,2]=np.ravel(imagecropped)

def twoD_GaussianV2(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xdata_tuple                                                        
    xo = float(xo)                                                              
    yo = float(yo)                                                              
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)   
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)    
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)   
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)         
                        + c*((y-yo)**2)))                                   
    return g
#%%
#Do the fitting
popt2, pcov2 = opt.curve_fit(twoD_GaussianV2,(xyzarr[:,0],xyzarr[:,1]), xyzarr[:,2], p0=initial_guess,maxfev = 10000)

#Create array for the fit, compared to the other method
xyzarrfit=np.zeros((x.size*y.size,3))
xyzarrfit[:,0]=np.repeat(x,y.size)
xyzarrfit[:,1]=np.tile(y,x.size)
xyzarrfit[:,2]=twoD_GaussianV2([xyzarrfit[:,0],xyzarrfit[:,1]],*popt2) #Note this is popt2

xyzarrfit2=np.zeros((x.size*y.size,3))
xyzarrfit2[:,0]=np.repeat(x,y.size)
xyzarrfit2[:,1]=np.tile(y,x.size)
xyzarrfit2[:,2]=twoD_GaussianV2([xyzarrfit[:,0],xyzarrfit[:,1]],*popt) #Note this is popt

#Some plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(xyzarr[:,0],xyzarr[:,1],xyzarr[:,2],'g.')

plt.plot(xyzarrfit[:,0],xyzarrfit[:,1],xyzarrfit[:,2],'b.')
plt.plot(xyzarrfit[:,0],xyzarrfit[:,1],xyzarrfit2[:,2],'r.')
#%%
np.meshgrid(x,y)
#%%
#Some assorted plotting etc
def f(x, y):
    return twoD_GaussianV2([x,y],*popt2)


X, Y = np.meshgrid(x, y)
Z = f(X, Y)
X = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#ax.plot_wireframe(X[0], X[1], imagecropped,color='k')
#Compare how close both method are to the actual result
ax.plot_wireframe(X[0], X[1], data_fitted.reshape(ylen, xlen)-imagecropped,color='r')
ax.plot_surface(X[0], X[1], Z-imagecropped,color='g')
