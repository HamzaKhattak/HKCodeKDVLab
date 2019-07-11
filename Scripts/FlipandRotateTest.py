import sys, os


CodeDR="F:\TrentDrive\Research\KDVLabCode\HKCodeKDVLab"
dataDR="F:\TrentDrive\Research\Droplet forces film gradients\SlideData2"


os.chdir(CodeDR)
sys.path.append('./Tools')
import DropletprofileFitter as df
sys.path.remove('./Tools')
print(os.getcwd())


os.chdir(dataDR)

print(df.flipper)