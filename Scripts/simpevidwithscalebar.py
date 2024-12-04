#%% This cell is for only the video
import matplotlib.animation as animation
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.pyplot as plt
import tifffile as tf
import numpy as np
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",'font.size': 16,
})

imgloc = 'returnlarge2_MMStack_Pos0.ome.tif'

pixsize = 2.25e-6
insecperframe = 0.5
xrealtime = 10
inFPS = 1/insecperframe
outputFPS = inFPS*xrealtime


#Import the images of interest and a base image for background subtraction
tifobj = tf.TiffFile(imgloc)
numFrames = len(tifobj.pages)
ims =  tf.imread(imgloc,key=slice(0,numFrames))

#finalframe = params['endframe']
finalframe = len(ims)


dim = ims.shape[1:]
dimr = dim[1]/dim[0]
# ax refers to the axis propertis of the figure
fig, ax = plt.subplots(1,1,figsize=(8,8/dimr))

#line, = ax.plot([], [], lw=2)
im=ax.imshow(ims[0],cmap='gray',aspect='equal')

ax.axis('off')
ax.get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
ax.get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis

#txt = ax[0].text(.90, .95, 'B={x:.2f}G'.format(x=gaussvals[0]),fontsize=12, ha='center',transform=plt.gca().transAxes)
scalebar = ScaleBar(pixsize,frameon=False,location='lower right',pad=1.5) # 1 pixel = 0.2 meter
ax.add_artist(scalebar)


#plt.tight_layout(pad=-4)
# animation function.  This is called sequentially
def animate_func(i):
	im.set_array(ims[i])
	return im,
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
anim.save('returndrops.mp4',writer=writer,dpi=200)