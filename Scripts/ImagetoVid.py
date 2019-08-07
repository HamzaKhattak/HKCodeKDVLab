'''
This code is for quickly combining files for later analysis and then saving a video to make for easy playing
'''
import os
#Specify where the data is and where plots will be saved
dataDR=r"E:\Newtips\SpeedAnalysis"
os.chdir(dataDR) #Set  current working direcotry to where the files are
os.system("ffmpeg -r 1/200 -i *.tif -c:v libx264 -vf fps=25 -pix_fmt yuv420p out.mp4")