'''
This code performs the edge location and cross correlation analysis across multiple images
'''

import sys, os, glob, pickle, re
import matplotlib.pyplot as plt
import numpy as np
import importlib
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter

#import similaritymeasures

#%%
#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"E:\SoftnessTest\SIS3per14wt1"


os.chdir(CodeDR) #Set  current working direcotry to the code directory


sys.path.append('./Tools') #Add the tools to the system path so modules can be imported

#Import required modules
import DropletprofileFitter as df
importlib.reload(df)
import Crosscorrelation as crco
importlib.reload(crco)
import ImportTools as ito 
importlib.reload(ito)
import EdgeDetection as ede
importlib.reload(ede)
import PlateauAnalysis as planl
importlib.reload(planl)

#Remove to avoid cluttering path
sys.path.remove('./Tools') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)

#%%
folderpaths, foldernames, dropProp = ito.foldergen(os.getcwd())

dropProp = [np.load(i+'DropProps.npy') for i in folderpaths]

exparams = np.genfromtxt('Aug27-SIS3per14wt1.csv', dtype=float, delimiter=',', names=True) 

springc = 0.155 #N/m
mperpix = 0.75e-6 #meters per pixel

#%%
#These are all in descending order of speed, so reverse to match dropProps
varr=exparams[r"Speed_ums"][::-1]
tsteps=exparams[r"Time_per_frame_required"][::-1]
eedistance=exparams[r"Distance_um"][::-1]
numcycles=exparams[r"Number_of_periods"][::-1]
indexorder=[i for i in range(varr.size)] #If want in another order for some reason
#%%

'''
tsteps = [13,2.6,.5,1.3,0.65,.5,.48]
varr = [.1,.5,10,1,2,5,8]
indexorder=[2,6,5,4,3,1,0]
'''
timebeforestop=[2*numcycles[i]*eedistance[i]/varr[i] for i in range(len(varr))]

labelarr=['$%.1f \mu m /s$' %i for i in varr]

def tarrf(arr,tstep):
	'''
	Simply returns a time array for plotting
	'''
	return np.linspace(0,len(arr)*tstep,len(arr)) 

colorarr=plt.cm.jet(np.linspace(0,1,len(tsteps)))
timearr=[tarrf(dropProp[i][:,0],tsteps[i]) for i in range(len(tsteps))]


#%%
forcedat = np.array([arr[:,0] * mperpix * springc for arr in dropProp])

lengthdat = np.array([planl.anglefilter(arr[:,2]-arr[:,1]) * mperpix for arr in dropProp])

langledat=np.array([planl.anglefilter(arr[:,5]) for arr in dropProp])
rangledat=np.array([planl.anglefilter(arr[:,6]) for arr in dropProp])

forceplateaudata=[None]*len(indexorder)
angleplateaudata=[None]*len(indexorder)

for i in indexorder:
	#Get force plateau data
	plateaudata=planl.plateaufilter(timearr[i],forcedat[i],[30,timebeforestop[i]],smoothparams=[2,1],sdevlims=[.1,1],outlierparam=2)
	topdata=plateaudata[4][0]
	bottomdata=plateaudata[4][1]
	#Indexes for other steady state values
	idh=plateaudata[5][0]
	idl=plateaudata[5][1]
	
	#Use indexes to get angle data
	filteredAngles = [ planl.smoothingfilter(np.abs(langledat[i])), planl.smoothingfilter(np.abs(rangledat[i])) ] 
	
	#Create arrays of time  angles to organize better
	tArrscut = [timearr[i][idh],timearr[i][idh],timearr[i][idl],timearr[i][idl]]
	anglesCut= [filteredAngles[0][idh],filteredAngles[1][idh],filteredAngles[0][idl],filteredAngles[1][idl]]
	filteredAngles = [np.transpose([tArrscut[i],anglesCut[i]]) for i in range(4)]
	
	#force cluster analysis
	ftopm = planl.clusteranalysis(topdata,30)
	fbotm = planl.clusteranalysis(bottomdata,30)
	'''
	#angle means for high forces
	atopm1 = planl.clusteranalysis(filteredAngles[0],3)
	atopm2 = planl.clusteranalysis(filteredAngles[1],3)
	
	#angle means fow low forces
	abotm1 = planl.clusteranalysis(filteredAngles[2],3)
	abotm2 = planl.clusteranalysis(filteredAngles[3],3)
	'''

	
	forceplateaudata[i] = [[topdata,bottomdata,idh,idl],ftopm,fbotm]
	#angleplateaudata[i] = [filteredAngles,atopm1,atopm2,abotm1,abotm2]


'''
#allleading=[np.concatenate([arr[0][0],arr[0][3]]) for arr in angleplateaudata]
#alltrailing=[np.concatenate([arr[0][1],arr[0][2]]) for arr in angleplateaudata]
lsimplemeans = [np.mean(arr[:,1]) for arr in allleading]
lsimpleerr = [np.std(arr[:,1]) for arr in allleading]
tsimplemeans = [np.mean(arr[:,1]) for arr in alltrailing]
tsimpleerr = [np.std(arr[:,1]) for arr in alltrailing]

'''

#Get forces in nice form
forceav=np.array([1e6*(arr[1][0][0]-arr[2][0][0])/2 for arr in forceplateaudata])
errbars=np.array([1e6*np.sqrt((arr[1][0][1]**2+arr[2][0][1]**2))/2 for arr in forceplateaudata])


anglemeans = [ [arr[1][0][0],arr[2][0][0],arr[3][0][0],arr[4][0][0]] for arr in angleplateaudata]
anglemeans=np.array(anglemeans)
anglestds = [ [arr[1][0][1],arr[2][0][1],arr[3][0][1],arr[4][0][1]] for arr in angleplateaudata]
anglestds=np.array(anglestds)



#%%
gs = gridspec.GridSpec(3, 1)



fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0]) 
'''
for i in indexorder:
	ax1.plot(timearr[i]*varr[i],forcedat[i]*1e6,label=labelarr[i],color=colorarr[i])
	ax2.plot(timearr[i]*varr[i],lengthdat[i]*1e6,color=colorarr[i])
	ax3.plot(timearr[i]*varr[i],langledat[i],color=colorarr[i])
	ax3.plot(timearr[i]*varr[i],rangledat[i],color=colorarr[i])
ax1.legend()
'''

for i in [0]:
	ax1.plot(timearr[i]*varr[i],forcedat[i]*1e6)
	ax2.plot(timearr[i]*varr[i],lengthdat[i]*1e6)
	ax3.plot(timearr[i]*varr[i],langledat[i])
	ax3.plot(timearr[i]*varr[i],rangledat[i])

ax1.set_ylabel('Pipette F ($\mu N$)')

ax2.set_ylabel('Droplet length ($\mu m$)')

ax3.set_ylim(45,95)
ax3.set_ylabel('Contact angle')
ax3.set_xlabel('Approx Substrate distance travelled ($\mu m$)')

plt.tight_layout()
#%%
gs = gridspec.GridSpec(2, 1)
fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])



ax1.errorbar(varr,forceav,yerr=errbars,fmt='.')
for i in [0,3]:
	ax2.errorbar(varr,anglemeans[:,i],yerr=anglestds[:,i],fmt='r.')

for i in [1,2]:
	ax2.errorbar(varr,anglemeans[:,i],yerr=anglestds[:,i],fmt='b.')


ax2.set_xlabel(r"Speed ($\mu m/s$)")
ax1.set_ylabel(r"Force ($px$)")
ax2.set_ylabel(r"Angle")
#%%
runName=os.path.basename(os.getcwd())

forcevt=np.column_stack([varr,forceav,errbars,anglemeans,anglestds])
np.save(runName+'pvveldat.npy',forcevt)
#%%


