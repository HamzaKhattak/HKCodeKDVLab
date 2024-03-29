# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:36:22 2022

@author: WORKSTATION
"""
import numpy as np
import pickle
from skimage.io import imread as imread2
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
runparams = np.loadtxt('runsparams.csv',skiprows=1,dtype=str,delimiter=',')
run_names = runparams[:,0]
run_time_steps = runparams[:,1].astype(float)
run_crops = runparams[:,2:].astype(float)
run_crops = run_crops.astype(int)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

which_im = 7
xmax = -1

#Get the crop points from the run parameters
x0=run_crops[which_im][0]
x1=run_crops[which_im][2]
y0=run_crops[which_im][1]
y1=run_crops[which_im][3]
halfpoint = run_crops[which_im][4]
allimages = imread2(run_names[which_im])[:xmax,y0:y1,x0:x1]
leadtxt = run_names[which_im].split('.')[0]

with open(leadtxt+'edges', "rb") as input_file:
	edges = pickle.load(input_file)

edges=edges[:xmax]
dat = np.load(leadtxt+'.npy')
lins = np.load(leadtxt+'lin.npy')



upper_line_params=lins[0]
lower_line_params=lins[1]

xfinal = dat[0][:xmax]
seps = dat[3][:xmax]
yfinal = dat[1][:xmax]
speeds = np.abs(np.gradient(xfinal))

def quickfilter2(x,y):
	params = np.polyfit(x, y,2)
	polytrend = np.polyval(params,x)
	b, a = butter(1,.05)
	smoothed = filtfilt(b,a,y-polytrend)
	smoothy = polytrend+smoothed
	return x, smoothy

pos, smoothspeeds = quickfilter2(seps, speeds)

pixsize = 1.78e-6 #pixel size of camera in m
numRuns = len(run_names)




from matplotlib import animation
from matplotlib_scalebar.scalebar import ScaleBar




xarray=np.arange(allimages.shape[2])

# ax refers to the axis propertis of the figure
fig, ax = plt.subplots(2,1,figsize=(8,6), gridspec_kw={'height_ratios': [1, 1]})
im = ax[1].imshow(allimages[0],cmap=plt.cm.gray,aspect='equal')
im.set_clim(0, 256) #if want to reproduce original image rather than full scale
scalebar = ScaleBar(pixsize,frameon=False,location='lower right',font_properties={'size':26},pad=1.5) # 1 pixel = 0.2 meter
#im.set_clim(0, 256) #if want to reproduce original image rather than full scale

ax[0].plot(seps*2,speeds*2,'.')
ax[0].plot(seps*2,smoothspeeds*2,'k-')
ax[0].set_xlabel('Separation distance ($\mathrm{\mu m}$)')
ax[0].set_ylabel('speed ($\mathrm{\mu m s^{-1}}$)')

ax[1].axis('off')
ax[1].get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
ax[1].get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis


edgeline, = ax[1].plot(edges[0][:,0],edges[0][:,1],color='red',marker='.',linestyle='',markersize=1,animated=True)

topline, = ax[1].plot(xarray,np.poly1d(upper_line_params[0])(xarray),color='cyan',marker='.',linestyle='',markersize=1,animated=True)
botline, = ax[1].plot(xarray,np.poly1d(lower_line_params[0])(xarray),color='cyan',marker='.',linestyle='',markersize=1,animated=True)

currentspeed, = ax[0].plot([], [],'ro', animated=True)
centerpoint, =  ax[1].plot([], [],'ro', animated=True)





def init():
	"""
	This function gets passed to FuncAnimation.
	It initializes the plot axes
	"""
	#Set plot limits etc
	ax[1].add_artist(scalebar)


	#plt.tight_layout()
	return centerpoint,edgeline,currentspeed,topline,botline,
#fig.tight_layout(pad=0)

def update_plot(it):
	#global xAnim, yAnim
	
	#This this section plots the force over time
	edgeline.set_data(edges[it][:,0],edges[it][:,1])
	topline.set_data(xarray,np.poly1d(upper_line_params[it])(xarray))
	botline.set_data(xarray,np.poly1d(lower_line_params[it])(xarray))
	centerpoint.set_data(xfinal[it],yfinal[it])
	currentspeed.set_data(seps[it]*2,speeds[it]*2)
	#Plot of image
	im.set_data(allimages[it])
	
	return centerpoint,im,edgeline,currentspeed,topline,botline,
plt.tight_layout()

#Can control which parts are animated with the frames, interval is the speed of the animation
# now run the loop
ani = animation.FuncAnimation(fig, update_plot, frames=np.arange(0,len(allimages)), interval=1,
                    init_func=init, repeat_delay=1000, blit=True)


#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

plt.show()

#%%
print(np.mean(dat[2])*180/np.pi)
#%%
'''
Writer = animation.writers['ffmpeg']
writer = Writer(fps=10,extra_args=['-vcodec', 'libx264'])
ani.save('speedtracking.mp4',writer=writer,dpi=200)
'''
#%%



