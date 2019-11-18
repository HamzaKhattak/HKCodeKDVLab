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
dataDR=r"E:\SoftnessTest\SISThickness2\01um_1"


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
mperpix = 0.75e-6 #meters per pixel
#%%
#Get the image sequence imported
x1c=270
x2c=1300
y1c=310
y2c=940
croppoints=[x1c,x2c,y1c,y2c]
croppoints=[x1c,x2c,y1c,y2c]
allimages=ito.omestackimport(dataDR)
allimages=ito.cropper(allimages,*croppoints)



#%%
edges=ito.openlistnp(os.path.join(dataDR,'edgedata.npy'))
dropprops = ito.openlistnp(os.path.join(dataDR,'allDropProps.npy'))
AnglevtArray, EndptvtArray, ParamArrat, rotateinfo = dropprops

#rotate to match image
edges=[df.rotator(arr,rotateinfo[0],rotateinfo[1][0],rotateinfo[1][1]) for arr in edges]

centrepos, loaddat=np.load("correlationdata.npy")

inputxvt=loaddat[:,:,0]
inputyvt=loaddat[:,:,1]
#%%
fig = plt.figure(figsize=(4,3))
plt.plot(-allimages[0][500],'k-')
plt.plot(-allimages[300][500],'b-')
plt.xlabel('x value')
plt.ylabel('Intensity')
plt.tight_layout()
#%%
fig = plt.figure(figsize=(4,3))
plt.plot(inputxvt[300],inputyvt[300],'b-')
plt.xlim(-150,100)
plt.axvline(centrepos[300,0],c='r')
plt.xlabel('Shift (pixels)')
plt.ylabel('Similarity')
plt.tight_layout()
#%%
dt=13.6
endT=yAnim[:,0].size*dt #The total time over the data, assumes equal spacing

yAnim=inputxvt
xAnim=inputyvt
tArray=np.linspace(0,endT,yAnim[:,0].size) #Make a list of times


# ax refers to the axis propertis of the figure
fig, ax = plt.subplots(2,1,figsize=(5,5))
im = ax[0].imshow(allimages[0],cmap=plt.cm.gray,aspect='equal')
scalebar = ScaleBar(0.75e-6,frameon=False,location='lower right') # 1 pixel = 0.2 meter

ax[0].axis('off')
ax[0].get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
ax[0].get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis

edgeline, = ax[0].plot(edges[0][:,0],edges[0][:,1],color='cyan',marker='.',linestyle='',markersize=1,animated=True)
fitline, = ax[0].plot(edges[0][:,0],edges[0][:,1],'ro',markersize=2,animated=True)
line, =  plt.plot([], [],'b-', animated=True)
vline2, = ax[1].plot([],'go',animated=True)

#Create fitted data
xvals=np.arange(-20,20)
yvals=np.arange(-20,20)


#im1=ax[0].imshow(allimages[0],cmap=plt.cm.gray,aspect='equal',alpha=0.5)
#p=ax[1].axvline(x=centrepos[0,0])
forcecorrdat=yAnim[:,:]*springc*mperpix*1e6
forcecendat=centrepos[:,0]*springc*mperpix*1e6
midforceval=(np.max(forcecendat)+np.min(forcecendat))/2

plottimeconv=1/3600
startcutframe=50 #Which frame to start on
	
def init():
	"""
	This function gets passed to FuncAnimation.
	It initializes the plot axes
	"""
	#Set plot limits etc
	ax[0].add_artist(scalebar)
	ax[0].set_xlim(100, 1000)
	ax[0].set_ylim(100, 500)
	#ax[0].set_ylim(allimages[0,:,1].min, allimages[0,:,1].max)
	'''
	#Use this section to plot cross correlation
	ax[1].set_xlim(-15, 15)
	ax[1].set_ylim(-.1, 1)
	ax[1].set_xlabel('Force ($\mu N$)')
	ax[1].set_ylabel('Cross correlation value')
	'''
	#Use this section to plot force over time
	ax[1].set_xlim(0, endT*plottimeconv) #convert to hrs if needed
	ax[1].set_ylim(-5, 5)
	ax[1].set_xlabel('time (hrs)')
	ax[1].set_ylabel('Force ($\mathrm{\mu N}$)')


	plt.tight_layout()
	return line,edgeline,fitline,vline2,
fig.tight_layout(pad=0)
#need number of timesteps total
nt=yAnim[:,0].size
def update_plot(it):
	global xAnim, yAnim
	#Plot of force data
	'''
	#This section plots the cross correlation
	line.set_data(forcecorrdat[it,:]-midforceval, xAnim[it,:])
	vline2.set_data([[forcecendat[it]-midforceval,forcecendat[it]-midforceval],[-.1,1]])
	'''
	
	#This this section plots the force over time
	line.set_data((tArray[:it]-tArray[startcutframe])*plottimeconv, forcecendat[:it]-midforceval)
	vline2.set_data([(tArray[it]-tArray[startcutframe])*plottimeconv, forcecendat[it]-midforceval])
	
	
	#Plot of image
	im.set_data(allimages[it])
	
	#Plot of lines over image
	edgeline.set_data([edges[it][:,0],edges[it][:,1]])
	#initialize the c and y values
	xvals=np.arange(0,40)
	yvals=np.arange(0,40)
	#This is to account for the flipped fitting done (or else cant fit vertical)
	#yvals=df.pol2ndorder(xvals,*ParamArrat[it][1])
	xvals=df.pol2ndorder(yvals,*ParamArrat[it][1])
	yvals=yvals+EndptvtArray[it,1,1]
	xvals=xvals+EndptvtArray[it,1,0]
	comboarr=np.transpose([xvals,yvals])
	yvals = df.rotator(comboarr,rotateinfo[0],rotateinfo[1][0],rotateinfo[1][1])
	fitline.set_data([yvals[:,0],yvals[:,1]])
	
	return line,im,edgeline,fitline,vline2,
plt.tight_layout()

#Can control which parts are animated with the frames, interval is the speed of the animation
# now run the loop
ani = animation.FuncAnimation(fig, update_plot, frames=np.arange(startcutframe,tArray.size,2), interval=20,
                    init_func=init, repeat_delay=1000, blit=True)


#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

plt.show()

#%%
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30,extra_args=['-vcodec', 'libx264'])
file_path=r'C:\Users\WORKSTATION\Dropbox\FigTransfer\Symposium Day'
file_path=os.path.join(file_path,'AnimatedExperimentv2.mp4')
ani.save(file_path,writer=writer,dpi=200)