'''
Script for plotting a time series line
Authors:Hamza Khattak
'''

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import importlib
import sys, os
#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"E:\DualAngles\FirstSpeedScan"


os.chdir(CodeDR) #Set  current working direcotry to the code directory


sys.path.append('./Tools') #Add the tools to the system path so modules can be imported

#Import required modules
import DropletprofileFitter as df
importlib.reload(df)
import ImportTools as ito 
importlib.reload(ito)
import EdgeDetection as ede
importlib.reload(ede)
import Crosscorrelation as crco
importlib.reload(crco)

from matplotlib_scalebar.scalebar import ScaleBar

#Remove to avoid cluttering path
sys.path.remove('./Tools') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)
#%%
springc = 0.155 #N/m
mperpixside = 0.75e-6 #meters per pixel
mperpixtop = 0.75e-6
#%%
selectfolder='0p1ums0'
edgeparams = ito.openlistnp('edgedetectparams.npy')

imfold = os.path.join(dataDR,selectfolder)
#Get the image sequence imported

cropside,sideimaparam,croptop,topimaparam = edgeparams

#importimage
impath = ito.getimpath(selectfolder)
imseq = ito.fullseqimport(impath)

#Seperate out side and top views
sidestack = imseq[:,0]
topstack = imseq[:,1]

sidestack = ito.cropper2(sidestack,cropside)
topstack = ito.cropper2(topstack,croptop)

#%%
plt.imshow(sidestack[150],plt.cm.gray)
plt.figure()
plt.imshow(topstack[150],plt.cm.gray)
#%%

#Edges
sideedges = ito.openlistnp(os.path.join(imfold,'sideedgedata.npy'))
topedges =  ito.openlistnp(os.path.join(imfold,'topedgedata.npy'))

#Calculated properties
dropprops = ito.openlistnp(os.path.join(imfold,'allDropProps.npy'))
extractDat = np.load(os.path.join(imfold,'DropProps.npy'))
centrepos, loaddat = np.load(os.path.join(imfold,"correlationdata.npy"),allow_pickle=True)

#Need to rotate the edges
AnglevtArray, EndptvtArray, ParamArrat, rotateinfo = dropprops[0]

topcxvals = dropprops[1][1]
#rotate to match image
#sideedges=[df.rotator(arr,rotateinfo[0],rotateinfo[1][0],rotateinfo[1][1]) for arr in sideedges]



#Top edge and fit tracking
plt.imshow(sidestack[50],cmap=plt.cm.gray)
plt.plot(sideedges[50][:,0],sideedges[50][:,1],'.')
plt.figure()
plt.imshow(topstack[50],cmap=plt.cm.gray)
plt.plot(topedges[50][:,0],topedges[50][:,1],'.')
#%%
print(ParamArrat[0][1].shape)
#%%
plt.plot(EndptvtArray[:,1,0])
#%%
tVals = extractDat[0]
forceVals = extractDat[1]
perimVals = extractDat[-2]
#topstack, topedges, sidestack and sideedges

#Get the fit of the contact angle
#This is to account for the flipped fitting done (or else cant fit vertical)
#yvals=df.pol2ndorder(xvals,*ParamArrat[it][1])
cafit = [None]*len(tVals)
for i in range(len(tVals)):
	xvals=np.arange(0,100)
	yvals=np.arange(0,100)
	xvals=df.pol2ndorder(yvals,*ParamArrat[i][1])
	yvals=yvals+EndptvtArray[i,1,1]
	xvals=xvals+EndptvtArray[i,1,0]
	comboarr=np.transpose([xvals,yvals])
	yvals = df.rotator(comboarr,rotateinfo[0],rotateinfo[1][0],rotateinfo[1][1])
	cafit[i] = np.array([yvals[:,0],yvals[:,1]])


# ax refers to the axis propertis of the figure
fig, axs = plt.subplots(2,2,figsize=(8,4))

#Side edges and cross correlation forces

imside = axs[0,0].imshow(sidestack[0],cmap=plt.cm.gray,aspect='equal') 
sidescalebar = ScaleBar(0.75e-6,frameon=False,location='lower right') # 1 pixel = 0.2 meter

axs[0,0].axis('off')
axs[0,0].get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
axs[0,0].get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis

sideedgeline, = axs[0,0].plot(sideedges[0][:,0],sideedges[0][:,1],color='cyan',marker='.',linestyle='',markersize=1,animated=True)
sidefitline, = axs[0,0].plot(cafit[0][0],cafit[0][1],color='red',marker='.',linestyle='',markersize=1,animated=True)


#Re add the side fit line for contact angles
forceline, = axs[1,0].plot([],marker='.',linestyle='',markersize=1,animated=True)
forcedat=centrepos[:,0]*springc*mperpixside*1e6

#Top view and fits
imtop = axs[0,1].imshow(topstack[0],cmap=plt.cm.gray,aspect='equal') 
topedgeline, = axs[0,1].plot(topedges[0][:,0],topedges[0][:,1],color='cyan',marker='.',linestyle='',markersize=1,animated=True)
topfitline, = axs[0,1].plot(topcxvals[0][:,0],topcxvals[0][:,1],color='red',marker='.',linestyle='',markersize=1,animated=True)

topscalebar = ScaleBar(0.75e-6,frameon=False,location='lower right') # 1 pixel = 0.2 meter

axs[0,1].axis('off')
axs[0,1].get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
axs[0,1].get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis

#Re add the side fit line for contact angles
perimline, = axs[1,1].plot([],marker='.',linestyle='',markersize=1,animated=True)
perimdat=perimVals*mperpixside*1e6

	
def init():
	"""
	This function gets passed to FuncAnimation.
	It initializes the plot axes
	"""
	#Set plot limits etc
	axs[0,0].add_artist(sidescalebar)
	#axs[0,0].set_ylim(0, 500)
	#Use this section to plot force over time
	axs[1,0].set_xlim(0, tVals[-1]) #convert to hrs if needed
	axs[1,0].set_ylim(-100, 100)
	axs[1,0].set_xlabel('time (s)')
	axs[1,0].set_ylabel('Force ($\mathrm{\mu N}$)')
	
	#Set plot limits etc
	axs[0,1].add_artist(topscalebar)
	#axs[0,0].set_ylim(0, 500)
	#Use this section to plot force over time
	axs[1,1].set_xlim(0, tVals[-1]) #convert to hrs if needed
	axs[1,1].set_ylim(600, 700)
	axs[1,1].set_xlabel('time (s)')
	axs[1,1].set_ylabel('Perimeter')
	
	
	plt.tight_layout()
	return sideedgeline,sidefitline,forceline,topedgeline,perimline,topfitline

#fig.tight_layout(pad=0)
#need number of timesteps total
nt=len(tVals)
def update_plot(it):
	#global xAnim, yAnim
	#Image and fit
	imside.set_data(sidestack[it])
	sideedgeline.set_data([sideedges[it][:,0],sideedges[it][:,1]])
	sidefitline.set_data([cafit[it][0],cafit[it][1]])
	
	#This this section plots the force over time
	forceline.set_data([tVals[:it], forcedat[:it]])
	
	imtop.set_data(topstack[it])
	topedgeline.set_data([topedges[it][:,0],topedges[it][:,1]])
	topfitline.set_data([topcxvals[it][0],topcxvals[it][1]])
	perimline.set_data([tVals[:it], perimdat[:it]])
	return imside,sideedgeline,sidefitline,forceline,perimline,imtop,topedgeline,topfitline,
#plt.tight_layout()

#Can control which parts are animated with the frames, interval is the speed of the animation
# now run the loop
ani = animation.FuncAnimation(fig, update_plot, frames=range(len(tVals)), interval=20,
                    init_func=init, repeat_delay=1000, blit=True)


#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

plt.show()

#%%
np.save('testingcirclefit.npy',topstackedges[62])

#%%
plt.plot(perimdat,'.')

#%%
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30,extra_args=['-vcodec'•, 'libx264'])
file_path=r'C:\Users\WORKSTATION\Dropbox\FigTransfer\Sept14'
file_path=os.path.join(file_path,'AnimatedExperimentdualfixed.mp4')
ani.save(file_path,writer=writer,dpi=200)