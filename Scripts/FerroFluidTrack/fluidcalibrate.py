# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 13:00:14 2024

@author: WORKSTATION
"""

import imageio
import matplotlib.pyplot as plt
import tifffile as tf
import cv2 as cv
import numpy as np
from scipy import signal

from scipy.optimize import curve_fit

#%%
#Import the images of interest and a base image for background subtraction
tifobj = tf.TiffFile('run_MMStack_Pos0.ome.tif')
numFrames = len(tifobj.pages)
ims =  tf.imread('run_MMStack_Pos0.ome.tif',key=slice(0,numFrames))
#%%
plt.imshow(ims[0],cmap='gray')




#%%
'''

'''





def cropper(inarray,crops):
	'''
	Simply crops a template given crop points in [x1,y1],[x2,y2] form
	'''
	return inarray[:,crops[0][1]:crops[1][1],crops[0][0]:crops[1][0]]



leftcrop = [[600,200],[750,500]]
rightcrop = [[910,200],[1110,500]]
left = cropper(ims,leftcrop)
right = cropper(ims,rightcrop)
plt.imshow(right[0])
#%%


def gauss(x, H, A, x0, sigma): 
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))



#%%
def normify(x):
	r = -x
	r = r-np.min(r)
	r=r/np.max(r)
	return r


def findpipcenters(cropped):
	h, w = croppedim.shape
	locs = np.zeros(h,2)
	blurred = cv.blur(croppedim,(20,20))
	x = np.arange(w)
	for i in range(h):
		y = normify(blurred[i])
		maxloc = np.argmax(y)
		po, pcov = curve_fit(gauss, x, y,p0=[0,1,maxloc,20])
		locs[i] = po[2]
	return locs

def findshifted(ims,croppoints):
	cropped = cropper(ims,croppoints)
	y = np.arange(croppoints[0][1],croppoints[1][1])
	xs = np.zeros((ims.shape[0],cropped.shape[1]))
	for i in range(len(ims)):
		locs = findpipcenter(cropped[i])
		xs[i] = locs+croppoints[0][0]
	return y, xs



def findcents(x,y,cutpoint):
	'''
	Extends a line from the fitted points to find the center location of the droplet
	Given a location to cutoff for the droplet
	'''
	fitparams = np.polyfit(y,x,1)
	print(fitparams)
	x0 = np.polyval(np.poly1d(fitparams), cutpoint)
	return cutpoint,x0



testl = findshifted(ims[:10],pip1crop)
testr = findshifted(ims[:10],pip2crop)


dropcenterloc = findcents(testl[1][0],testl[0],650)

plt.plot(testl[1][0],testl[0])
plt.plot(testr[1][0],testr[0])
plt.plot(dropcenterloc[1],dropcenterloc[0],'ro')
plt.imshow(ims[0],cmap='gray')
#%%
y = normify(blurred[57])
maxloc = np.argmax(y)
po, pcov = curve_fit(gauss, x, y,p0=[0,1,maxloc,20])


plt.plot(normify(left[57]))
plt.plot(normify(blurred[57]))
plt.plot(y)
plt.plot(x,gauss(x,*po))
plt.axvline(po[2],color='k')
#%%


#plt.imshow(ims[0][200:400,850:1200])

def cropper(inarray,crops):
	'''
	Simply crops a template given crop points in [x1,y1],[x2,y2] form
	'''
	return inarray[:,crops[0][1]:crops[1][1],crops[0][0]:crops[1][0]]




pip1cropped = cropper(ims,pip1crop)
pip2cropped = cropper(ims,pip2crop)

plt.imshow(pip2cropped[0],cmap='gray')
plt.imshow(pip2cropped[500],alpha=.3)

#%%

#%%



def ccor(a,b,meth='cv.TM_CCOEFF_NORMED'):
	'''
	This code runs cross correlation on an image with a given template and mask
	The returned cross correlation is taken to the power of 3 and then 
	normalized to emphasize the peaks
	'''
	#Normalize the input vectors
	norma = (a - np.mean(a)) / (np.std(a))
	normb = (b - np.mean(b)) / (np.std(b))
	w, h = a.shape[::-1]
	match = signal.correlate(norma, normb, mode='full', method='auto')
	match=match/np.max(match)
	#Getting shifts
	#corrx = np.arange(2*norma.shape[1]-1)-(norma.shape[1]-1)
	#corry = np.arange(2*norma.shape[0]-1)-(norma.shape[0]-1)
	shiftx = norma.shape[1]-1
	shifty = norma.shape[0]-1
	return shiftx,shifty,match



def findmaxloc(im):
	#Returns the index of the maximum point in an array
	return np.unravel_index(im.argmax(), im.shape)



def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

def refinelocations(inputccor,windowsize):
	#Create the meshgrid
	initiallocs = findmaxloc(inputccor)
	x = np.linspace(-windowsize,windowsize,2*windowsize+1,dtype=int)
	y = x
	X = np.meshgrid(x, y)

	yc = initiallocs[0]
	xc = initiallocs[1]
	
	cropped = inputccor[yc-windowsize:yc+windowsize+1 , xc-windowsize:xc+windowsize+1]
	#initial_guess = (cropped[windowsize,windowsize],windowsize,windowsize-1,windowsize-1,windowsize,0,cropped[0,0])
	initial_guess = (1,0,0,10,100,0,0)
	inputdata = np.ravel(cropped)
	
	popt, pcov = curve_fit(twoD_Gaussian,X,inputdata,p0=initial_guess,maxfev=1000)

	locs = popt[2]+yc,popt[1]+xc
	return locs


def getshifts(croppedims):
	shiftarray=np.zeros((len(croppedims),2))
	for i in range(len(croppedims)):
		sx,sy,corrim1 = ccor(croppedims[i],croppedims[0])
		refineloc = refinelocations(corrim1,10)
		shiftarray[i] = [refineloc[0]-sy,refineloc[1]-sx]
	return shiftarray


leftshifts = getshifts(pip1cropped)
rightshifts =  getshifts(pip2cropped)
#%%
plt.plot(leftshifts[:,1]-rightshifts[:,1])
#%%
	

sx,sy,corrim1 = ccor(pip1cropped[0],pip1cropped[300])
refineloc = refinelocations(corrim1,10)
totalshift = 
print(refineloc[1]-sx)
#%%


rx,ry = [refineloc[0],refineloc[1]]
plt.imshow(ims[0],cmap='gray')
#plt.plot(rx+pip1crop[0][0],ry++pip1crop[0][1],'ro')
plt.plot(pip1crop[0][0],pip1crop[0][1],'ro')

plt.plot(rx+pip1crop[0][0],ry+pip1crop[0][1],'bo')

plt.plot(pip1crop[1][0],pip1crop[1][1],'ro')
#%%

x,y,corrim2 = ccor(testim1,testim2)

test2 =  refinelocations(corrim2,10)



plt.imshow(corrim1,cmap='gray')
plt.plot(test[1],test[0],'ro')
#plt.imshow(corrim2,alpha=.3)
#%%

import matplotlib.animation as animation





fig, ax = plt.subplots()

ax.plot(ims[0][500],'b-')
ax.set_xlim(600,1000)
line, = ax.plot(ims[0][500],'r-')


def animate(i):
	ax.plot(ims[0][500],'r-')
	line.set_ydata(ims[i][500])  # update the data.
	return line,


ani = animation.FuncAnimation(
    fig, animate, frames = len(ims), interval=20, blit=True,repeat=False)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

plt.show()
#%%
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30,extra_args=['-vcodec', 'libx264'])
ani.save('samplevid.mp4',writer=writer,dpi=200)