# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:43:44 2023

@author: hamza
"""

from pymeasure.instruments.hp import HP33120A
import numpy as np
import time
HP = HP33120A("ASRL5")
#%%
HP.write('SYSTem:REMote')
#%%

freqlist = np.linspace(1000,2000,num=100)
testfreq = [format(x, '.2E') for x in freqlist]

voltlist = np.linspace(0.1,0.5,num=100)
testvolt = [format(x, '.3E') for x in voltlist]
#%%
for i in testfreq:
	time.sleep(.1)
	stringtowrite = "FREQ " + i
	#HP.write("FREQ 5.0E+3")
	HP.write(stringtowrite)
#%%
for i in testvolt:
	time.sleep(.1)
	stringtowrite = "VOLT " + i
	#HP.write("FREQ 5.0E+3")
	HP.write(stringtowrite)
#%%

'''
Note page 141 of the HP33120 manual, there are glitches at certain voltages
and the output drops to zero momentarily at other voltages
Means this may not be the best for voltage ramping
'''

#%%
freqlist = np.linspace(100,500,num=1000)

HP.write("FREQ 1E+3")
HP.write("VOLT 5.05E-2")
#%%
errs = HP.ask('SYSTem:ERRor?')
