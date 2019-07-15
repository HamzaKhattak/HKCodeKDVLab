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
CodeDR="F:\TrentDrive\Research\KDVLabCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR="F:\TrentDrive\Research\Droplet forces film gradients\SlideData2"


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
allimages=ito.stackimport(dataDR+"\Translate1ums5xob.tif")
allimages=ede.cropper(allimages,9,750,715,898)
loaddat=np.load("datcorr.npy")
centrepos=np.load("centerloc.npy")
#Array containing, extract the y and x values first (always at x=0)
#Input array of the form of a list of yvalues for every timestep
#ie [[x1(t0),x2(t0),x3(t0)],[x1(t1),x2(t1),x3(t1)]]
inputxvt=loaddat[:,:,0]
inputyvt=loaddat[:,:,1]
#%%
endT=10 #The total time over the data, assumes equal spacing
yAnim=inputxvt
xAnim=inputyvt
tArray=np.linspace(0,endT,yAnim[:,0].size) #Make a list of times


# ax refers to the axis propertis of the figure
fig, ax = plt.subplots(2,1,figsize=(8,8))
line, =  plt.plot([], [],'r-', animated=True)
vline2, = ax[1].plot([],animated=True)

im= ax[0].imshow(allimages[0],cmap=plt.cm.gray,aspect='equal')
im1=ax[0].imshow(allimages[0],cmap=plt.cm.gray,aspect='equal',alpha=0.5)
#p=ax[1].axvline(x=centrepos[0,0])

def init():
    """
    This function gets passed to FuncAnimation.
    It initializes the plot axes
    """
    #Set plot limits etc

    ax[0].set_xlim(0, 741)
    ax[1].set_xlim(-681, 60)
    ax[1].set_ylim(-.1, 1)
    ax[1].set_xlabel('shift (pixels)')
    ax[1].set_ylabel('Cross correlation value')
    plt.tight_layout()
    return line,vline2,
#need number of timesteps total
nt=yAnim[:,0].size
def update_plot(it):
    global xAnim, yAnim
    line.set_data(yAnim[it,:], xAnim[it,:])
    im.set_data(allimages[it])
    vline2.set_data([[centrepos[it,0],centrepos[it,0]],[-.1,1]])
    return line,im,im1,vline2,
plt.tight_layout()

#Can control which parts are animated with the frames, interval is the speed of the animation
# now run the loop
ani = animation.FuncAnimation(fig, update_plot, frames=np.arange(0,tArray.size,1), interval=5,
                    init_func=init, repeat_delay=1000, blit=True)


#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

plt.show()
#%%
plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg'
FFwriter = animation.FFMpegWriter(fps=50,extra_args=['-vcodec', 'libx264'])
ani.save('basic_animation.mp4', writer = FFwriter )
#%%
Writer = animation.writers['ffmpeg']
writer = Writer(fps=60)
ani.save('basic_anim.mp4',writer=writer,dpi=100)

#%%
print(animation.writers)