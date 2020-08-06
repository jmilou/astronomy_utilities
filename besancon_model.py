#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:32:23 2020

@author: jmilli
"""

import os
import numpy as np
import pandas as pd

path_data = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')

def read_Kband_cumulative_counts():
    """
    Read the K2 table from https://model.obs-besancon.fr/modele_starcounts.php
    that returns  the cumulative star counts per square degree in different directions, 
    that is up to a given magnitude between 3 and 21. The indicated number at the 
    top of each column is the maximum magnitude to which the stars have been counted. 
    Hence column N7 gives counts with K<7
    """
    model = pd.read_csv(os.path.join(path_data,'Besancon_model_K_band_cumulative_star_counts.txt'),delimiter=' ')
    model.drop(columns='Avtot',inplace=True) 
    model.rename(columns={'N*': 'N10'},inplace=True)
    model['cos_latitude'] = np.cos(np.deg2rad(model['latitude']))
    model['sin_latitude'] = np.sin(np.deg2rad(model['latitude']))
    model['cos_longitude'] = np.cos(np.deg2rad(model['longitude']))
    model['sin_longitude'] = np.sin(np.deg2rad(model['longitude']))
    return model
    
def get_star_count_below_Kmag(longitude,latitude,Kmaglimit,fov):
    """
    Input:
        - longitude in deg
        - latitude in deg
        - Kmaglimit: limiting magnitude in the K band
        - fov in arcsec^2
    """
    if Kmaglimit<4 or Kmaglimit>=21:
        print('K magnitude outside limits: {0:.2f}'.format(Kmaglimit))
        return
    Kmag_inf = int(Kmaglimit)
    Kmag_sup = Kmag_inf+1
    key_inf = 'N{0:d}'.format(Kmag_inf)    
    key_sup = 'N{0:d}'.format(Kmag_sup)    
    model = read_Kband_cumulative_counts()
    delta_latitude = (np.deg2rad(model['latitude']-latitude))
    delta_longitude = (np.deg2rad(model['longitude']-longitude))

    cos_latitude = np.cos(np.deg2rad(latitude))
    # sin_latitude = np.sin(np.deg2rad(latitude))
    # this formula is inaccurate (numerical accuracy problem, see https://en.wikipedia.org/wiki/Great-circle_distance)
    # delta_angle = np.arccos(model['sin_latitude']*sin_latitude+model['cos_latitude']*cos_latitude*cos_delta_latitude)
    delta_angle = 2*np.arcsin(np.sqrt(np.sin(delta_latitude/2)**2+model['cos_latitude']*cos_latitude*np.sin(delta_longitude/2)**2))
    # print(np.rad2deg(delta_angle))
    # delta_angle = np.mod(np.rad2deg(delta_angle),360)
    argmin = np.argmin(np.abs(np.asarray(delta_angle)))
    print('Nearest point for longitude={0:.0f} and latitude={1:.0f}'.format(\
            model['longitude'][argmin],model['latitude'][argmin],\
            np.abs(delta_angle[argmin])))
    star_count_inf = float(model[key_inf][argmin])
    star_count_sup = float(model[key_sup][argmin])
    star_count_per_square_degree = star_count_inf+(star_count_sup-star_count_inf)/1.*(Kmaglimit-Kmag_inf)
    star_count_per_square_arcsec = star_count_per_square_degree/(60.*60.)**2
    return star_count_per_square_arcsec*fov

if __name__ == '__main__':
    test = get_star_count_below_Kmag(3.,-1.,14,12**2)
    print(test)    
    print(get_star_count_below_Kmag(211.65658083333335,-15.715350555555554,20.1,12**2))
