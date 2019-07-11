'''
Script for plotting a time series line
Authors:Hamza Khattak
'''

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
#%%
#Array containing, extract the y and x values first (always at x=0)
#Input array of the form of a list of yvalues for every timestep
#ie [[x1(t0),x2(t0),x3(t0)],[x1(t1),x2(t1),x3(t1)]]
inputxvt=np.load("Dat1.npy")
#Input array of the form of a list of xvalues for every timestep
inputyvt=np.load("Dat2.npy")
#%%
endT=10 The total time over the data, assumes equal spacing
yAnim=inputxvt
xAnim=inputyvt
tArray=np.linspace(0,endT,yAnim[:,0].size) #Make a list of times


# ax refers to the axis propertis of the figure
fig, ax = plt.subplots(figsize=(14,14))
line, =  plt.plot([], [],'r.', animated=True)


def init():
    """
    This function gets passed to FuncAnimation.
    It initializes the plot axes
    """
    #Set plot limits etc
    ax.set_xlim(-2, 2)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    plt.tight_layout()
    return line,
#need number of timesteps total
nt=yAnim[:,0].size
def update_plot(it):
    global xAnim, yAnim
    line.set_data(yAnim[it,:], xAnim[it,:])
    return circ2,circ,rect,line,
plt.tight_layout()

#Can control which parts are animated with the frames, interval is the speed of the animation
# now run the loop
ani = animation.FuncAnimation(fig, update_plot, frames=np.arange(1500,tArray.size-2000,1), interval=2,
                    init_func=init, repeat_delay=1000, blit=True)


#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

plt.show()
#%%
plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg'
FFwriter = animation.FFMpegWriter(fps=100,extra_args=['-vcodec', 'libx264'])
ani.save('basic_animation.mp4', writer = FFwriter )
