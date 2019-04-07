#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 10:34:33 2019

@author: jmilli
"""

from astropy.io import ascii,fits
import numpy as np
import os #,sys
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
from astropy import units as u
from astropy import constants as const
from astropy.modeling.blackbody import blackbody_lambda, blackbody_nu

path = os.path.join(os.path.dirname(os.path.abspath(__file__)) ,'data')


df_filter_V = pd.read_csv(os.path.join(path,'passband_filter_V_T.csv'))
df_filter_VBB = pd.read_csv(os.path.join(path,'passband_filter_V_T.csv'))

df_filter_VBB = pd.read_csv(os.path.join(path,'passband_filter_VBB_Zimpol.csv'))

#df_filter_VBB.rename(columns={0: 'lambda(nm)',\
#                1:'lambda*R(lambda)'}, inplace=True)        
#
interp_function_transmission = interp1d(df_filter_V['lambda(nm)'],\
                                        df_filter_V['lambda*R(lambda)'],\
                                        bounds_error=False,fill_value=0)
interpolated_transmission_VT = interp_function_transmission(df_filter_VBB['lambda(nm)'])

plt.plot(df_filter_VBB['lambda(nm)'],interpolated_transmission_VT,color='blue')                                
plt.plot(df_filter_V['lambda(nm)'],df_filter_V['lambda*R(lambda)'],color='red')  
plt.plot(df_filter_VBB['lambda(nm)'],df_filter_VBB['lambda*R(lambda)'],color='green')                                
plt.grid()
#plt.xlim(450,950)


wave_set = np.asarray(df_filter_VBB['lambda(nm)'])*10*u.AA
freq_set = (const.c/wave_set).to(u.Hertz)


temperature = 9730 * u.K
lammin=400
lammax=950
#waveset = np.linspace(lammin,lammax,lammax-lammin+1)*10 * u.AA
#freqset = (const.c/waveset).to(u.Hertz)

with np.errstate(all='ignore'):
#    flux = blackbody_lambda(waveset, temperature)
    flux = blackbody_nu(freq_set, temperature)


fig, ax = plt.subplots(figsize=(8,5))
ax.plot(df_filter_VBB['lambda(nm)'], flux.value)
#ax.axvline(wavemax.value, ls='--')
ax.get_yaxis().get_major_formatter().set_powerlimits((0, 1))
ax.set_xlabel(r'$\lambda$ ({0})'.format(wave_set.unit))
ax.set_ylabel(r'$B_{\nu}(T)$')
ax.set_title('Blackbody, T = {0}'.format(temperature))
ax.grid()


plt.plot(df_filter_VBB['lambda(nm)'],interpolated_transmission_VT,color='blue')                                
plt.plot(df_filter_V['lambda(nm)'],df_filter_V['lambda*R(lambda)'],color='red')  
plt.plot(df_filter_VBB['lambda(nm)'],df_filter_VBB['lambda*R(lambda)'],color='green')                                
plt.plot(df_filter_VBB['lambda(nm)'], flux.value/np.max(flux.value))
plt.grid()


star_flux_V_theory_Jy = 18.18  # 18.18 Jy (+ 18.33 - 18.03)
scaling_factor = star_flux_V_theory_Jy/(np.sum(interpolated_transmission_VT*flux.value)/np.sum(interpolated_transmission_VT))
scaled_spectrum = scaling_factor*flux.value

flux_star_V = np.sum(scaled_spectrum*interpolated_transmission_VT)/np.sum(interpolated_transmission_VT)
flux_star_VBB = np.sum(scaled_spectrum*df_filter_VBB['lambda*R(lambda)'])/np.sum(df_filter_VBB['lambda*R(lambda)'])

print(flux_star_V)
print(flux_star_VBB)

#18.18
#16.13 (+16.25 -15.99) error bar: 0.14Jy
