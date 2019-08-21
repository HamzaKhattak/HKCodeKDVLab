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
dataDR=r"E:\SpeedScan\5umreturn_1"


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


#Remove to avoid cluttering path
sys.path.remove('./Tools') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)
#%%

#%%

x1c=616
x2c=1500
y1c=500
y2c=855
croppoints=[x1c,x2c,y1c,y2c]
croppoints=[x1c,x2c,y1c,y2c]
allimages=ito.folderstackimport(dataDR)
allimages=ito.cropper(allimages,*croppoints)
#%%
edges=ito.openlistnp(os.path.join(dataDR,'edgedata.npy'))
dropprops = ito.openlistnp(os.path.join(dataDR,'allDropProps.npy'))
AnglevtArray, EndptvtArray, ParamArrat, rotateinfo = dropprops

#rotate to match image
edges=[df.rotator(arr,rotateinfo[0],rotateinfo[1][0],rotateinfo[1][1]) for arr in edges]

loaddat=np.load("CCorall.npy")
centrepos=np.load("Ccorcents.npy")

inputxvt=loaddat[:,:,0]
inputyvt=loaddat[:,:,1]

#%%
endT=10 #The total time over the data, assumes equal spacing
yAnim=inputxvt
xAnim=inputyvt
tArray=np.linspace(0,endT,yAnim[:,0].size) #Make a list of times


# ax refers to the axis propertis of the figure
fig, ax = plt.subplots(2,1,figsize=(8,8))
im = ax[0].imshow(allimages[0],cmap=plt.cm.gray,aspect='equal')

edgeline, = ax[0].plot(edges[0][:,0],edges[0][:,1],color='cyan',marker='.',linestyle='',markersize=1,animated=True)
fitline, = ax[0].plot(edges[0][:,0],edges[0][:,1],'r-',markersize=2,animated=True)
line, =  plt.plot([], [],'r-', animated=True)
vline2, = ax[1].plot([],animated=True)


xvals=np.arange(0,20)
#im1=ax[0].imshow(allimages[0],cmap=plt.cm.gray,aspect='equal',alpha=0.5)
#p=ax[1].axvline(x=centrepos[0,0])

def init():
    """
    This function gets passed to FuncAnimation.
    It initializes the plot axes
    """
    #Set plot limits etc

    ax[0].set_xlim(0, 800)
    #ax[0].set_ylim(allimages[0,:,1].min, allimages[0,:,1].max)
    ax[1].set_xlim(-100, 100)
    ax[1].set_ylim(-.1, 1)
    ax[1].set_xlabel('shift (pixels)')
    ax[1].set_ylabel('Cross correlation value')
    plt.tight_layout()
    return line,edgeline,fitline,vline2,
plt.tight_layout()
#need number of timesteps total
nt=yAnim[:,0].size
def update_plot(it):
	global xAnim, yAnim
	line.set_data(yAnim[it,:], xAnim[it,:])
	im.set_data(allimages[it])
	edgeline.set_data([edges[it][:,0],edges[it][:,1]])
	yvals=df.pol2ndorder(xvals,*ParamArrat[it][0])
	yvals=yvals+EndptvtArray[it,0,1]
	comboarr=np.transpose([xvals+EndptvtArray[it,0,0],yvals])
	yvals = df.rotator(comboarr,rotateinfo[0],rotateinfo[1][0],rotateinfo[1][1])
	fitline.set_data([comboarr[:,0],comboarr[:,1]])
	vline2.set_data([[centrepos[it,0],centrepos[it,0]],[-.1,1]])
	return line,im,edgeline,fitline,vline2,
#plt.tight_layout()

#Can control which parts are animated with the frames, interval is the speed of the animation
# now run the loop
ani = animation.FuncAnimation(fig, update_plot, frames=np.arange(0,tArray.size,1), interval=50,
                    init_func=init, repeat_delay=1000, blit=True)


#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

plt.show()

#%%
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30,extra_args=['-vcodec', 'libx264'])
ani.save('basic_anim.mp4',writer=writer,dpi=100)