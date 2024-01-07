# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 23:58:38 2023

@author: WORKSTATION
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 10:09:12 2023

@author: WORKSTATION
"""


import os, glob, imageio, re
#need to install imageio and imagio-ffmpeg and tkinter 
import tifffile as tf
import numpy as np
import ndtiff as ndt
from datetime import datetime
from matplotlib import pyplot as plt
file_path = '2percentrun1_MMStack_Pos0.ome.tif'



fields_path = "percent2size1run1.csv"
fields = np.loadtxt(fields_path,delimiter=',')

tifobj = tf.TiffFile(file_path)
numFrames = len(tifobj.pages)
imageframes =  tf.imread(file_path,key=slice(0,numFrames))

metdat = tifobj.imagej_metadata
maininfo = metdat['Info']
st = 'StartTime'
starttime = re.findall('"StartTime": "(.+?)"', maininfo)[0]
starttime = datetime.strptime(starttime,'%Y-%m-%d %H:%M:%S.%f %z')
epochtime = starttime.timestamp()
omexml = tifobj.ome_metadata
keyword1 = 'TimeIncrement='
match1 = re.findall(f"{keyword1}.*?(\d+[.]\d+)", omexml)[0]
keyword2 = 'TimeIncrementUnit='
match2 = re.findall(f'{keyword2}"(.+?)" ', omexml)[0]



if match2=='ms':
	timestep = float(match1)/1000
if match2 == 's':
	timestep = float(match1)

#in case there is an error in the metadata, fall back on manual timestep
inFPS = str(1/timestep)

plt.plot(fields[:,0],fields[:,1],label='gaussmeter')

#quick check frame 106 is where split happens ~38 Gauss
x = np.linspace(epochtime,epochtime+.5*numFrames,num=numFrames)
plt.axvline(x[218])
plt.legend()