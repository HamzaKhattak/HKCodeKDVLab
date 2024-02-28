# -*- coding: utf-8 -*-
"""
This code outputs droplet positions (unorganized) from a video
This code is meant to run in Spyder so you can zoom in to 
"""

import numpy as np
import matplotlib.pyplot as plt



import imageio, os, importlib, sys, time


from matplotlib import colors
from win11toast import notify
import tifffile as tf

import requests
#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"F:\ferro\Experiments\Concentration05\Pip3\multidrop4_1"

#Use telegram to notify
tokenloc = r"F:\ferro\token.txt"


os.chdir(CodeDR) #Set  current working direcotry to the code directory


sys.path.append('./Scripts/FerroFluidTrack') #Add the tools to the system path so modules can be imported

#Import required modules
import PointFindFunctions as pff
importlib.reload(pff)

import FrametoTimeAndField as fieldfind
importlib.reload(fieldfind)

import NNfindFunctions as nnfind
importlib.reload(nnfind)

#Remove to avoid cluttering path
sys.path.remove('./Scripts/FerroFluidTrack') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)


#%%
params = pff.openparams('InputRunParams.txt')
run_name = params['run_name']
fieldfind.findGuassVals(params['fieldspath'], params['inputimage'],params['GaussSave'])

#Import the images of interest and a base image for background subtraction
tifobj = tf.TiffFile(params['inputimage'])
numFrames = len(tifobj.pages)
ims =  tf.imread(params['inputimage'],key=slice(0,numFrames))
background = tf.imread(params['backgroundim'])



#Run the image correction to flatten the brighness
correctedims = pff.imagepreprocess(ims, background)
#%%
plt.figure()
plt.imshow(correctedims[0],cmap='gray') #Imshow to allow cropping to find template crop locations
plt.figure()
plt.imshow(correctedims[1300],cmap='gray')
#%%
params = pff.openparams('InputRunParams.txt')
'''
Get the masks used in cross correlation
'''


numTemplates = params['numtemplates']




#templatemetadata = {'crops': crops,'maskthresholds': mask_thresholds,'ccorthresh': ccorr_thresholds,'minD': [ccminsep,compareminsep]}


c_white = colors.colorConverter.to_rgba('red',alpha = 0)
c_red= colors.colorConverter.to_rgba('red',alpha = .1)
cmap_rb = colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white,c_red],512)
#plt.imshow(thresholdfinder(correctedimage),cmap=cmap_rb)

templates = [None]*numTemplates
masks = [None] * numTemplates
for i in range(numTemplates):
	templates[i] = pff.templatecropper(correctedims[params['cropframes'][i]],params['crops'][i])
	masks[i] = templates[i] < params['mask_thresholds'][i]
	masks[i]=masks[i].astype(np.float32)
	plt.figure()	
	plt.imshow(templates[i],cmap='gray')
	plt.imshow(masks[i],cmap=cmap_rb)
#%%
'''
Run the analysis on some test images to make sure it works
'''
params = pff.openparams('InputRunParams.txt')
inims = correctedims[params['testframes']]

testpos,testrpos = pff.fullpositionfind(inims, templates, masks, params,combinebytemplate=False)

plt.figure()
for j in testrpos[0]:
	plt.plot(j[:,1],j[:,0],'.')
plt.imshow(inims[0],cmap='gray')


plt.figure()
for j in testrpos[1]:
	plt.plot(j[:,1],j[:,0],'.')
plt.imshow(inims[1],cmap='gray')



#%%
'''
Run the analysis and save the relevant metadata
'''
allpositions, allrefinedpositions = pff.fullpositionfind(correctedims, templates, masks, params, reportfreq=50)


pff.savelistnp(run_name+'positions.pik',allpositions)

notifytext = run_name + ' is done.'
notify(notifytext)
#also notify through telegram
#send notification through telegram
with open(tokenloc) as f:
	tegnot = f.read()
	
token, chatid = tegnot.split('\n')
url = f"https://api.telegram.org/bot{token}"
params = {"chat_id": chatid, "text": "The run be complete"}
r = requests.get(url + "/sendMessage", params=params)
#%% This cell is for running the video with the images and nearest neightbours

import matplotlib.animation as animation
from matplotlib_scalebar.scalebar import ScaleBar


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",'font.size': 12,
})

gaussvals = np.loadtxt('FrametoGauss.csv',delimiter=',')[:,2]
pixsize = 2.25e-6

#allpositions = pff.openlistnp(run_name+'positions.pik')
allpositions = pff.openlistnp('testsmoothpositions.pik')
nns = nnfind.findstrings(allpositions,params['dropletradius'])
nnindex,nnnum,nnfrac = nnfind.findNNforsequence(allpositions,params['dropletradius']+3)

fspos = [None]*len(nnnum)
othpos = [None]*len(nnnum)
zerpos = [None]*len(nnnum)
test = nnnum[0]<5

for i in range(len(nnnum)):
	fspos[i] = allpositions[i][np.logical_or(nnnum[i]==5,nnnum[i]==6)]
	othpos[i] = allpositions[i][(nnnum[i]<5)*(nnnum[i]>0)]
	zerpos[i] = allpositions[i][nnnum[i]==0]
		

finalframe = params['endframe']



fig,ax = plt.subplots(1,2,figsize=(12,7),gridspec_kw={'width_ratios': [2, 1]})
#line, = ax.plot([], [], lw=2)
im=ax[0].imshow(correctedims[0],cmap='gray',aspect='equal')

#points, = ax.plot(allrefinedlocs[0][:,1],allrefinedlocs[0][:,0],'.')
#points, = ax[0].plot(allpositions[0][:,1],allpositions[0][:,0],'r.',markersize=2)

fpoints, = ax[0].plot(fspos[0][:,1],fspos[0][:,0],'o',markersize=2,color='lightsteelblue')
opoints, = ax[0].plot(othpos[0][:,1],othpos[0][:,0],'o',markersize=2,color='orange')
zpoints, = ax[0].plot(zerpos[0][:,1],zerpos[0][:,0],'o',markersize=2,color='limegreen')



ax[0].axis('off')
ax[0].get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
ax[0].get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis

#txt = ax[0].text(.90, .95, 'B={x:.2f}G'.format(x=gaussvals[0]),fontsize=12, ha='center',transform=plt.gca().transAxes)
scalebar = ScaleBar(pixsize,frameon=False,location='lower right',font_properties={'size':12},pad=1.5) # 1 pixel = 0.2 meter
ax[0].add_artist(scalebar)


clust, = ax[1].plot(gaussvals[0],nns[0,0],'.',label = '5,6')
string, = ax[1].plot(gaussvals[0],nns[0,1],'.',label = '4-1')
disp, = ax[1].plot(gaussvals[0],nns[0,2],'.',label = '0')

ax[1].set_xlim(0,45)
ax[1].set_ylim(-.1,1.19)
ax[1].set_xlabel(r'$B \ \mathrm{(G)}$')
ax[1].set_ylabel(r'$f$')



asp = (np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0])*1.5
ax[1].set_aspect(asp)

ax[1].legend(title ='NN',loc = 'upper left')
fig.subplots_adjust(hspace=.05)
#plt.tight_layout(pad=-4)
# animation function.  This is called sequentially
def animate_func(i):
	im.set_array(correctedims[i])
	#points.set_data(allrefinedlocs[i][:,1],allrefinedlocs[i][:,0])
	#points.set_data(allpositions[i][:,1],allpositions[i][:,0])
	fpoints.set_data(fspos[i][:,1],fspos[i][:,0])
	zpoints.set_data(zerpos[i][:,1],zerpos[i][:,0])
	opoints.set_data(othpos[i][:,1],othpos[i][:,0])
	
	#points.set_data(test2.y[i],test2.x[i])
	#txt.set_text('B={x:.2f}G'.format(x=gaussvals[i]))
	clust.set_data(gaussvals[:i],nns[:i,0])
	string.set_data(gaussvals[:i],nns[:i,1])
	disp.set_data(gaussvals[:i],nns[:i,2])
	
	return im,clust,string,disp,fpoints,opoints,zpoints,

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = np.arange(0,finalframe,1),
                               interval = 1,blit=True, # in ms
                               )

#%% This cell is for only the video
import matplotlib.animation as animation
from matplotlib_scalebar.scalebar import ScaleBar


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",'font.size': 16,
})

pixsize = 2.25e-6
insecperframe = .25
xrealtime = 5
inFPS = 1/insecperframe
outputFPS = inFPS*xrealtime




#finalframe = params['endframe']
finalframe = len(ims)


dim = ims.shape[1:]
dimr = dim[1]/dim[0]
# ax refers to the axis propertis of the figure
fig, ax = plt.subplots(1,1,figsize=(8,8/dimr))

#line, = ax.plot([], [], lw=2)
im=ax.imshow(correctedims[0],cmap='gray',aspect='equal')

ax.axis('off')
ax.get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
ax.get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis

#txt = ax[0].text(.90, .95, 'B={x:.2f}G'.format(x=gaussvals[0]),fontsize=12, ha='center',transform=plt.gca().transAxes)
scalebar = ScaleBar(pixsize,frameon=False,location='lower right',pad=1.5) # 1 pixel = 0.2 meter
ax.add_artist(scalebar)

txt = ax.text(.90, .95, '$B={x:.1f} \, \mathrm{{G}}$'.format(x=gaussvals[0]), ha='center',transform=plt.gca().transAxes)


#plt.tight_layout(pad=-4)
# animation function.  This is called sequentially
def animate_func(i):
	im.set_array(correctedims[i])
	txt.set_text('$B={x:.1f} \, \mathrm{{G}}$'.format(x=gaussvals[i]))
	return im,txt,
plt.tight_layout()
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = np.arange(0,finalframe,1),
                               interval = insecperframe/1000/xrealtime,blit=True, # in ms
                               )

#%% This section of code is for saving the video
Writer = animation.writers['ffmpeg']
writer = Writer(fps=outputFPS,extra_args=['-vcodec', 'libx264'])
anim.save('maintrackvideosmoothed.mp4',writer=writer,dpi=200)