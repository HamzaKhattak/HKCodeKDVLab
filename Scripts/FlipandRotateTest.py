import sys, os

#Specify the location of the Tools folder
CodeDR="F:\TrentDrive\Research\KDVLabCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR="F:\TrentDrive\Research\Droplet forces film gradients\SlideData2"


os.chdir(CodeDR) #Set  current working direcotry to the code directory
sys.path.append('./Tools') #Add the tools to the system path so modules can be imported
import DropletprofileFitter as df #Import required modules
sys.path.remove('./Tools') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)
#%%