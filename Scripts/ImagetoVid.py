'''
This code is for quickly combining files for later analysis and then saving a video to make for easy playing
'''
import os
dataDR="E:/Newtips/vidcombotest"
inFile="1ums.tif"
outFile="1umsv3.mov"

VidConvArg=dataDR+inFile+","+dataDR+'outFile'
os.chdir(dataDR)
#Need to add fiji to path or similar for UNIX
#Or use the trick of adding a .bit file in path that leads to it something like fiji.bat with:
'''
@echo off
echo.
C:\Users\WORKSTATION\PortPrograms\Fiji.app\ImageJ-win64.exe %*
'''

#Create video from tif stack using Fiji, could also maybe load into numpy array, but would likely be slower
os.system("fiji --headless - macro ConvertToVid.ijm"+'\"'+VidConvArg+'\"')
#Compressed video
os.system("ffmpeg -i 1umsv3.mov -c:v libx264 -b:v 2M -maxrate 2M -bufsize 1M output.mp4 ")
