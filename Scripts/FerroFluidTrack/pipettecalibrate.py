# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

import imageio as io

import os, sys, importlib
import tifffile as tf
from scipy.optimize import curve_fit
#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"F:\ferro\air\waterdropcalib\pip1_4"

#Use telegram to notify
tokenloc = r"F:\ferro\token.txt"


os.chdir(CodeDR) #Set  current working direcotry to the code directory


sys.path.append('./Scripts/FerroFluidTrack') #Add the tools to the system path so modules can be imported

#Import required modules
import AirDropFunctions as adf
importlib.reload(adf)



import FrametoTimeAndField as ftf
importlib.reload(ftf)

#Remove to avoid cluttering path
sys.path.remove('./Scripts/FerroFluidTrack') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)
#%%
pixsize = 2.24e-6
imname = 'pip1_MMStack_Pos0.ome.tif'
tifobj = tf.TiffFile(imname)
numFrames = len(tifobj.pages)
ims =  tf.imread(imname,key=slice(0,numFrames))

plt.imshow(ims[0])
#%%

fig = plt.figure('Pick top left and bottom right corner and then fit lines')
plt.imshow(ims[0],cmap='gray')
plt.imshow(ims[-1],cmap='gray',alpha=0.3)
plt.grid(which='both')

print('Select crop points for droplet')
crop_points = np.floor(plt.ginput(2,timeout=200)) #format is [[xmin,ymin],[xmax,ymax]]
crop_points=crop_points.astype(int)

print('Select crop points for pipette')
pcrop_points = np.floor(plt.ginput(2,timeout=200)) #format is [[xmin,ymin],[xmax,ymax]]
pcrop_points=pcrop_points.astype(int)

dropims= adf.cropper(ims,crop_points)
pipims = adf.cropper(ims,pcrop_points)
plt.close()
#%%


from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.ndimage import rotate
def linefind(pipimage,x0,y0):
	'''
	Outputs lines representing the edge of two pipettes from an image of pipettes
	'''
	x1=x0
	x2=x0+pipimage.shape[1]
	xs = np.arange(x1,x2)
	
	topvals = np.zeros(pipimage.shape[1])
	botvals = np.zeros(pipimage.shape[1])
	for i in range(pipimage.shape[1]):
		flipped = np.max(pipimage[:,i])-pipimage[:,i]
		flippedsmooth = savgol_filter(flipped, 15, 3)
		diffs = np.abs(np.diff(flippedsmooth))
		maxdiff = np.max(diffs)
		peaklocs = find_peaks(diffs,height=.3*maxdiff,prominence=1,rel_height=.9)[0]
		topvals[i] = peaklocs[0]+y0
		botvals[i] = peaklocs[-1]+y0
	
	topfit = np.polyfit(xs,topvals, 1)
	botfit = np.polyfit(xs,botvals, 1)
	pipwidth = botfit[1]-topfit[1]
	centerline = np.mean([topfit,botfit],axis=0)
	return centerline, pipwidth




def pipettedef(dropimage,x0,y0):
	'''
	Outputs lines representing the edge of two pipettes from an image of pipettes
	'''
	x1=x0
	x2=x0+dropimage.shape[1]
	xs = np.arange(x1,x2)
	
	topvals = np.zeros(dropimage.shape[1])
	botvals = np.zeros(dropimage.shape[1])
	for i in range(dropimage.shape[1]):
		flipped = np.max(dropimage[:,i])-dropimage[:,i]
		flippedsmooth = savgol_filter(flipped, 11, 3)
		diffs = np.abs(np.diff(flippedsmooth))
		maxdiff = np.max(diffs)
		peaklocs = find_peaks(diffs,height=.3*maxdiff,prominence=1,rel_height=.9)[0]
		topvals[i] = peaklocs[0]+y0
		botvals[i] = peaklocs[-1]+y0
	
	return xs, topvals, botvals

def profilefind(pipim,dropim,crop_points,pcrop_points):
	'''
	Finds the profile of the droplet on the pipette but rotated to horizontal
	'''
	rotateparams, pipettewidth = linefind(pipim,pcrop_points[0,0],pcrop_points[0,1])
	#rotatedim = rotate(dropim,-np.arctan(rotateparams[0]))
	pipettelocs = pipettedef(dropim,crop_points[0,0],crop_points[0,1])
	samplex = np.arange(crop_points[0,0],crop_points[0,0]+dropim.shape[1])
	centered= [pipettelocs[2]-np.poly1d(rotateparams)(samplex),pipettelocs[1]-np.poly1d(rotateparams)(samplex)]
	dropprofile = np.mean(np.abs(centered),axis=0)
	return [samplex, pipettelocs, rotateparams, pipettewidth, dropprofile]
	







def volumeconvert(profile,pipwidth,pixsize):
	'''
	Converts a profile and pipettewidth to a volume
	'''
	totalvolume = np.sum(np.pi*profile**2)
	pipettevolume = len(profile)*np.pi*(pipwidth/2)**2
	dropvolume = totalvolume - pipettevolume
	return dropvolume*pixsize**3




def crosscorrelator(a,b):
	'''
	This function takes in two 1D arrays a and b, normalizes them
	to find the cross correlation of a with b and then returns and
	returns an [x,y] list with the index of 
	'''
	#Normalize the input vectors
	norma = (a - np.mean(a)) / (np.std(a) * len(a))
	normb = (b - np.mean(b)) / (np.std(b))
	#Use numpy correlate to find the correlation
	corry = np.correlate(norma, normb, 'full')
	#Shift back to find the x array
	corrx = np.arange(2*len(norma)-1)-(len(norma)-1)
	return np.transpose([corrx,corry])

def gaussfunc(x,a,mu,sig):
	return a*np.exp((-(x-mu)**2)/(2*sig))


def centerfinder(vecx,vecy,buff):
	'''
	This function takes a 1D vector and fits a gaussian to its max
	peak. The buff (an integer) argument decides how many points to use around the
	max value
	'''
	#Find where the max peak is generally
	maxpos=np.argmax(vecy)
	#Find the 2 edges to use for this fit and trim the data
	lefte=maxpos-buff
	righte=maxpos+buff
	xdata=vecx[lefte:righte]
	ydata=vecy[lefte:righte]
	#Perform the curve fit, guess parameters, just since maxpos
	# will be pretty close to the center
	popt, pcov = curve_fit(gaussfunc,xdata,ydata,p0=[1,vecx[maxpos],2*buff])
	#Find standard deviation in parameters
	perr = np.sqrt(np.diag(pcov))
	#Return parameter and standard deviation
	return popt, perr

def xvtfinder(images,baseimage,cutloc,gausspts1):
	'''
	Takes a image sequence and the original image and returns series of shifts
	as well as the full cross correlation arrays
	from the base image using cross correlation at the y pixel defined by cutloc
	gaussspts1 is the number of points to use in the gaussian fit on either side
	'''
	imdim=images.ndim
	#Account for single image case
	if imdim==2:
		images=np.expand_dims(images, 0)

	#Create empty arrays to store data
	centerloc=np.zeros([images.shape[0],2])
	alldat=np.zeros([images.shape[0],images.shape[1]*2-1,2])
	#autocorrelation for base
	basecut=baseimage[:,cutloc]
	basecorr=crosscorrelator(basecut,basecut)
	bgparam, bgerr = centerfinder(basecorr[:,0],basecorr[:,1],gausspts1)
	#Perform cross correlation and use gaussian fit to find center position
	for i in range(images.shape[0]):
		alldat[i] = crosscorrelator(images[i,:,cutloc],basecut)
		gparam, gerr = centerfinder(alldat[i,:,0],alldat[i,:,1],gausspts1)
		centerloc[i]=[gparam[1],gerr[1]]
	#Account for the 0 point
	centerloc = centerloc-[bgparam[1],0]
	return centerloc, alldat

#%%

samplex, pipettelocs, rotateparams, pipettewidth, dropprofile = profilefind(pipims[0],dropims[0],crop_points,pcrop_points)
plt.figure(figsize=(5,4))
plt.imshow(ims[0],cmap='gray')
plt.plot(pipettelocs[0],pipettelocs[1],'.',label = 'top')
plt.plot(pipettelocs[0],pipettelocs[2],'.',label = 'bottom')
plt.legend()
plt.plot(samplex,np.poly1d(rotateparams)(samplex),'r-') # the center line
print(pipettewidth)
print(str((volumeconvert(dropprofile,pipettewidth,pixsize))*10**12) +' nanoliters')
#%%
plt.figure(figsize=(5,4))


plt.plot(pipettelocs[2]-np.poly1d(rotateparams)(samplex))
plt.plot(pipettelocs[1]-np.poly1d(rotateparams)(samplex))
plt.plot(dropprofile)
#%%
volumes = np.zeros(len(dropims))
allparams = [None]*len(dropims)
for i in range(len(dropims)):
	allparams[i] = profilefind(pipims[i],dropims[i],crop_points,pcrop_points)
	samplex, pipettelocs, rotateparams, pipettewidth, dropprofile = allparams[i]
	volumes[i] = volumeconvert(dropprofile,pipettewidth,pixsize)
	
#%%
widths = [i[3] for i in allparams]
locs = [i[1] for i in allparams]
plt.plot(widths) 

#%%
for i in np.arange(0,len(ims),10):
	plt.plot(locs[i])
#%%


import matplotlib.animation as animation


from matplotlib_scalebar.scalebar import ScaleBar


fig, ax = plt.subplots(1,1)

im = ax.imshow(ims[0],cmap='gray')
ax.set_ylim(900,200)
ax.set_xlim(300,1300)
line, = ax.plot(locs[0][0],locs[0][1],'r.')
line2, = ax.plot(locs[0][0],locs[0][2],'r.')
scalebar = ScaleBar(pixsize,frameon=False,location='upper right',pad=0.5) 
ax.add_artist(scalebar)
ax.axis('off')


plt.tight_layout()
def init():
	"""
	This function gets passed to FuncAnimation.
	It initializes the plot axes
	"""
	#Set plot limits etc


	#plt.tight_layout()
	return 


def animate(i):
	im.set_data(ims[i])
	line.set_data(locs[i][0],locs[i][1])
	line2.set_data(locs[i][0],locs[i][2])
	#line.set_data(irightline[1][0]+rightshifts[i,1],irightline[0])
	return line,im,line2,


ani = animation.FuncAnimation(
    fig, animate, frames = len(ims), interval=1, blit=True,repeat=False)

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


#%%
correctedvolumes = volumes-np.mean(volumes[-50:]) #Subtract off final volume
masses = correctedvolumes*1000 #SI units for masses
forces = masses*9.81 #Convert mass to force
centers, alldats = xvtfinder(pipims,pipims[-1],0,10)

deflections = centers[:,0]*pixsize #center locations in real units
plt.plot(deflections*1e6,forces*1e9,'.') 
plt.xlabel(r'$d \ \mathrm{(\mu m)}$')
plt.ylabel(r'$F \ \mathrm{(n N)}$')
plt.tight_layout()

def ln(x,a):
	return a*x

forcenN = forces*1e9
deflectionmum = deflections*1e6

popt , pcov = curve_fit(ln, deflectionmum, forcenN)
print(popt)
xsamples = np.linspace(0,np.max(deflectionmum),num=500)
plt.plot(xsamples,ln(xsamples,*popt))

