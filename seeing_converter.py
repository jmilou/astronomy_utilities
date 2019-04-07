#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 10:22:28 2018

This module contains a list of functions that can be used to convert the IQ 
measured on an instrument into the seeing at a different wavelength, airmass...
It is based on the definition of the IQ as defined on the ESO webpage
https://www.eso.org/observing/etc/doc/helpkmos.html
by

FWHM_IQ = sqrt (FWHM_ATM^2 + FWHM_TEL^2 + FWHM_INS^2)	(1)

with 

FWHM_ATM = FWHM_ORANG  *  (λc/500)^(-1/5)  *  airmass^3/5 * Fcorr
where :
•	FWHM_ORANG is the current seeing measured by the MASS-DIMM at 500nm 
    and entered by the astronomer as an input in ORANG. 
•	airmass is the current airmass of the target 
•	λc is the observing wavelength in nm; this is bluest filter in the 
    OB science  templates.
•	Fcorr is  a correction factor defined as 
    (1 + FKolb * 2.182 * (r0/L0)^0.356)^0.5 where
    o	r0 is the Fried parameter: 
    r0 = 0.976 * 500E-09 / FWHM_ORANG * ((180/PI)*3600) * (λc/500)^1.2 * [airmass ^ (-3/5)] 
    o	Kolb factor: FKolb = 1/(1+300*D/L0)-1 such that FKolb(VLT) = -0.982
    o	L0 is the turbulence outer scale defined as  L0 = 46 m

FWHM_TEL is the telescope diffraction limit FWHM at the observing 
wavelength λc, for the VLT case:  FWHM_TEL = 0.000212 * λc/8.2  [arcsec] 

FWHM_INS (λc) is instrument transfer function FWHM at observing 
wavelength λc in arcsec taken from instrument.cf file of IP


@author: jmilli
"""

import numpy as np
from sympy.core.power import Pow
from sympy.solvers import nsolve
from sympy import Symbol

def convert_strehl(sr1,lambda1,lambda2):
    """
    Convert the strehl given at wavelength 1 to wavelenth 2
    Input:
        - sr1: Strehl ratio (between 0 and 1) at wavelength 1
        - lambda1: wavelength 1 (in the same unit as wavelength2)
        - lambda2: wavelength 2 (in the same unit as wavelength1)
    """
    return np.power(sr1,(lambda1/lambda2)**2)

def IQ2seeing(IQ1,wavelength1=1.2e-6,airmass1=1.,\
              wavelength2=500e-9,L0=20,D=8.,FWHMins=0.):
    """
    Converts the IQ measured at a given wavelength for a given airmass into 
    a seeing, taking the telescope and instrument transfer function into account
    as well as the atmosphere outer scale.
    Input:
        - IQ1: the image quality in arcsec
        - wavelength1: the wavelength in m at which the IQ is provided (by
                        default 1.2 micron)
        - wavelength2: the wavelength in m at which the seeing is desired
                    (by default 500nm)
        - L0: the turbulence outer scale (by default 20m)
        - airmass1: the airmass at which the IQ is measured
        - D: diameter of the telescope (8m by default)
        - FWHMins: the instrument transfer function in arcsec (by default 0)
    Output:
        - the seeing at zenith, at wavelength 2
    """
    FWHMtel = np.rad2deg(wavelength1/D)*3600
    FWHMatm = np.sqrt(IQ1**2-FWHMtel^2 - FWHMins^2)
    seeing = FWHMatm2seeing(FWHMatm,wavelength1=wavelength1,airmass1=airmass1\
                            ,wavelength2=wavelength2,L0=L0,D=D)
    return seeing
    
def FWHMatm2seeing(IQ1,wavelength1=1.2e-6,airmass1=1.,wavelength2=500e-9,\
                   L0=20,D=8.):
    """
    Converts the atmospheric FWHM measured at a given wavelength for a given 
    airmass into a seeing value at zenith, taking the turbulence outer 
    scale into account
    Input:
        - IQ1: the image quality in arcsec
        - wavelength1: the wavelength in m at which the IQ is provided (by
                        default 1.2 micron)
        - airmass1: the airmass at which the IQ is measured
        - D: diameter of the telescope (8m by default)
        - wavelength2: the wavelength in m at which the seeing is desired
                    (by default 500nm)
        - L0: the turbulence outer scale (by default 20m)
    Output:
        - the seeing at zenith, at wavelength 2
    """
    FKolb = 1./(1.+300*D/L0)-1 
    IQ_squared_rad = np.deg2rad(IQ1/3600.)**2
    coeffA = (0.9759 * wavelength2 * np.power(wavelength2/wavelength1,-1./5.)*\
        np.power(airmass1,3./5.))**2
    coeffB = coeffA*FKolb*2.182/np.power(L0,0.356)
    r0 = Symbol('r0',real=True)    
    bounds = (1.-3,2.)    
    r0_sol = nsolve(IQ_squared_rad*r0**2-coeffB*Pow(r0,0.356)-coeffA, bounds)
    if np.abs(np.imag(r0_sol))>1e-3:
        raise ValueError('Problem: the solver found a complex solution:',r0_sol)
    else:
        r0_sol_complex = np.complex(r0_sol)
        seeing2_arcsec = np.rad2deg(0.9759*wavelength2/r0_sol_complex.real)*3600.
    print('Input:')
    print('IQ: {0:.2f}", lambda: {1:4.0f}nm, airmass: {2:.2f}, outer scale: {3:.2f}m, D: {4:.1f}m'.format(\
          IQ1,wavelength1*1e9,airmass1,L0,D))
    print('Output:')
    print('lambda:{0:4.0f}nm, r0: {1:.2f}m, seeing: {2:.2f}"'.format(\
          wavelength2*1e9,r0_sol_complex.real,seeing2_arcsec))
    return seeing2_arcsec

def seeing2FWHMatm(seeing1,wavelength1=500.e-9,wavelength2=1200.e-9,L0=20,\
                   airmass2=1.,D=8.):
    """
    Converts the seeing at zenith at wavelength 1 into an image quality, taken 
    into account the airmass, turbulence outer scale
    Input:
        - wavelength1: the wavelength in m at which the seeing is provided (by
                        default 500nm)
        - airmass2: the airmass at which the IQ is desired
        - diameter D: diameter of the telescope (8m by default)
        - wavelength2: the wavelength in m at which the IQ is desired
                    (by default 1200nm)
        - L0: the turbulence outer scale (by default 20m)
    Output:
        - the IQ in arcsec
    """
    FKolb = 1./(1.+300*D/L0)-1 
    r0 = 0.9759 * wavelength1 / np.deg2rad(seeing1/3600.) * \
        np.power(wavelength1/500.e-9,1.2) # r0 at 500nm
    Fcorr = np.sqrt(1 + FKolb * 2.182 * np.power(r0/L0,0.356))
    FWHMatm = seeing1  *  np.power(wavelength2/wavelength1,-1./5.) * \
        np.power(airmass2,3./5.) * Fcorr    
    return FWHMatm   

if __name__ == '__main__':
    
#    seeing = FWHMatm2seeing(0.6,wavelength1=500e-9,wavelength2=500e-9,L0=20,\
#                   airmass1=1.,D=8.)
    
    IQ = seeing2FWHMatm(0.8,wavelength1=500.e-9,wavelength2=2.1e-6,L0=20,\
                   airmass2=1.06,D=8.)
    print(IQ)
