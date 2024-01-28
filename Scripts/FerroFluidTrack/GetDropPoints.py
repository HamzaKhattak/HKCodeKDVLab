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
dataDR=r"F:\ferro\Experiments\Concentration05\PipA1REplace\MultiDrop_2"

#Use telegram to notify
tokenloc = r"F:\ferro\token.txt"


os.chdir(CodeDR) #Set  current working direcotry to the code directory


sys.path.append('./Scripts/FerroFluidTrack') #Add the tools to the system path so modules can be imported

#Import required modules
import PointFindFunctions as pff
importlib.reload(pff)

import FrametoTimeAndField as fieldfind
importlib.reload(fieldfind)

#Remove to avoid cluttering path
sys.path.remove('./Scripts/FerroFluidTrack') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)


#%%
params = pff.openparams('InputRunParams.txt')
run_name = params['run_name']
fieldfind.findGuassVals(params['fieldspath'], params['inputimage'],params['GaussSave'])
#%%
#Import the images of interest and a base image for background subtraction
tifobj = tf.TiffFile(params['inputimage'])
numFrames = len(tifobj.pages)
ims =  tf.imread(params['inputimage'],key=slice(0,numFrames))
background = tf.imread(params['backgroundim'])

#%%

#Run the image correction to flatten the brighness
correctedims = pff.imagepreprocess(ims, background)
#%%
plt.figure()
plt.imshow(correctedims[0],cmap='gray') #Imshow to allow cropping to find template crop locations
plt.figure()
plt.imshow(correctedims[100],cmap='gray')
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
#%%
import matplotlib.animation as animation

gaussvals = np.loadtxt('FrametoGauss.csv',delimiter=',')[:,2]
pixsize = 2.25e-6
from matplotlib_scalebar.scalebar import ScaleBar
allpositions = pff.openlistnp(run_name+'positions.pik')





fig,ax = plt.subplots(figsize=(8,8))
#line, = ax.plot([], [], lw=2)
im=ax.imshow(correctedims[0],cmap='gray')

#points, = ax.plot(allrefinedlocs[0][:,1],allrefinedlocs[0][:,0],'.')
points, = ax.plot(allpositions[0][:,1],allpositions[0][:,0],'r.',markersize=2)



ax.axis('off')
ax.get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
ax.get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis

txt = ax.text(.90, .95, 'B={x:.2f}G'.format(x=gaussvals[0]),fontsize=12, ha='center',transform=plt.gca().transAxes)
scalebar = ScaleBar(pixsize,frameon=False,location='lower right',font_properties={'size':12},pad=1.5) # 1 pixel = 0.2 meter
ax.add_artist(scalebar)

def init():
	im.set_data(correctedims[0])
	ax.add_artist(scalebar)
	return im,points,

# animation function.  This is called sequentially
def animate_func(i):
	im.set_array(correctedims[i])
	#points.set_data(allrefinedlocs[i][:,1],allrefinedlocs[i][:,0])
	points.set_data(allpositions[i][:,1],allpositions[i][:,0])
	#points.set_data(test2.y[i],test2.x[i])
	txt.set_text('B={x:.2f}G'.format(x=gaussvals[i]))
	return im,points,txt,

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = len(correctedims),
                               interval = 1,blit=True, # in ms
                               )
#%%
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30,extra_args=['-vcodec', 'libx264'])
anim.save('samplevid.mp4',writer=writer,dpi=200)