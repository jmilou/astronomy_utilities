#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:22:00 2022

@author: millij
modified by Johan M with verbose option to avoid printing
modified by Vito S to allow vectorized queries, to resolve some targets not found in Gaia/2MASS/JC-based archives,
and to allow one to set a preferred naming convention 
"""

import numpy as np
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astropy.coordinates import name_resolve
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import ICRS, FK5 #,FK4, Galactic
from astropy.io import fits

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

import os
from astropy.table import Table
import re

def query_simbad_from_header(header,limit_G_mag=15,metadata=None,verbose=True):
    """
    Function similar to query_simbad, but using the date, coord and name extracted 
    from the header
    Input:
        - header: the header of a SPHERE file
    """
    if is_moving_object(header):
        print('The object is a moving target and no  information can be retrieved from Simbad')
        return None
    date = Time(h['DATE-OBS'])
    coords = SkyCoord(h['RA']*u.degree,h['DEC']*u.degree)
    name = h['OBJECT']
    return query_simbad(date,coords,name=name,limit_G_mag=limit_G_mag,metadata=metadata,verbose=verbose)


def get_best_id(simbad_table,pref_order):
    """
    Method not supposed to be used outside the query_simbad method
    Given a result table as returned from Simbad, identifies the best suited identifiers
    according to the rules specified by the user
    """
    
    best_ids = []
    n_st = len(simbad_table['IDS'])
    
    names = np.array([name.replace('NAME ','') for name in simbad_table['IDS'].value.data])
    default_names = np.array([name.replace('NAME ','') for name in simbad_table['MAIN_ID'].value.data])

    for i in range(n_st):
        id_list=np.array(names[i].split('|'))
        pref_i=0
        while pref_i<len(pref_order):
            if pref_order[pref_i]=='bayer_designation':
                j=0
                while j<len(id_list):
                    match = re.match('.*\s{1}[a-zA-Z]{3}$',id_list[j])
                    if match:
                        best_ids.append(match.group(0))
                        break
                    j+=1
                if match: break
                elif j==len(id_list): pref_i+=1
            elif pref_order[pref_i]=='proper_name':
                j=0
                while j<len(id_list):
                    match1 = re.match(".*[0-9]+.*", id_list[j])
                    match2 = re.match('.*\s{1}[a-zA-Z]{3}$',id_list[j])
                    match = re.match("[a-zA-z]{4,20}\s*[a-zA-z]*\s*[a-zA-z]*\s*",id_list[j])
                    break_cycle = False
                    if (type(match1)==type(None)) & (type(match2)==type(None)) & (type(match)!=type(None)):
                        best_ids.append(id_list[j])
                        break_cycle = True
                        break
                    j+=1
                if break_cycle: break
                elif j==len(id_list): pref_i+=1
            else:
                ww,=np.where(np.char.find(id_list,pref_order[pref_i])!=-1)
                if len(ww)>=1:
                    best_ids.append(id_list[ww[0]])
                    break
                else:
                    pref_i+=1
        if pref_i==len(pref_order): best_ids.append(default_names[i])

    return best_ids


def fix_binaries(table,objects,bin_flag,SimbadQuery):
    """
    Method not supposed to be used outside the query_simbad method
    Handles binary components that were not resolved by a Simbad query, trying to add the
    binary component flag to every identifier found for the full system to which
    the component belongs
    """

    w_bin,= np.where((bin_flag!='') & (table['MAIN_ID'].value==''))
    requery_names, which_star = [], []
    
    if len(w_bin)>0:
        for i in w_bin:
            results_table = Simbad.query_objectids(objects[i][:-1])
            if type(results_table)!=type(None):
                results = results_table['ID'].value.astype(str)
                requery_names.extend([n+bin_flag[i] for n in results])
                requery_names.extend([objects[i][:-1]])
                which_star.extend([objects[i]]*len(results))
                which_star.extend([objects[i][:-1]])
        which_star=np.array(which_star)
        new_search = SimbadQuery.query_objects(requery_names)
        for i in w_bin:
            ww,=np.where((which_star==objects[i]) & (new_search['MAIN_ID'].value!=''))
            if len(ww)>0:
                table[i]=new_search[ww[0]]
            elif bin_flag[i]=='A':
                ww,=np.where((which_star==objects[i][:-1]) & (new_search['MAIN_ID'].value!=''))
                if len(ww)>0:
                    table[i]=new_search[ww[0]]
                
    return table


def query_simbad(date,coords,name=None,limit_G_mag=15,metadata=None,force_cm=False,verbose = False,pref_order=['HIP','HD','HR']):
    """
    Function that tries to query Simbad to find the object. 
    It first tries to see if the star name (optional argument) is resolved 
    by Simbad. If not it searches for the pointed position (ra and
    dec) in a cone of radius 10 arcsec. If more than a star is detected, it 
    takes the closest from the (ra,dec).
    Input:
        - date: an astropy.time.Time object (e.g. date = Time(header['DATE-OBS'])
            If name is an array, date must be initialized as a Time object containing a list of dates.
        - name: a string (numpy array) with the name(s) of the source(s).
        - coords: a SkyCoord object. For instance, if we extract the keywords 
            of the fits files, we should use
            coords = SkyCoord(header['RA']*u.degree,header['DEC']*u.degree)
            SkyCoord('03h32m55.84496s -09d27m2.7312s', ICRS)
            If name is an array, coords must be initialized as a SkyCoord object containing a list of coordinates.
        - limit_G_mag: the limiting G magnitude beyond which we consider the star too 
            faint to be the correct target (optional, by default 15)
        - metadata: any additional information in the form of a dictionary that
            one wants to pass in the ouptut dictionary 
        - force_cm: if True, does not discard a cross-match that has no available  
            photometry in Simbad (optional, by default False)
        - pref_order: a list indicating the catalogues, in descending order, to be
            preferentially stored into the 'simbad_BEST_NAME' keyword. Examples of valid
            catalogues: 'HIP', 'Gaia DR3', 'Gaia DR2', '2MASS', 'TYC', 'HD', 'HR' etc.
            You can also use 'proper_name' to pick star names such as 'Sirius', or 'bayer_designation'
            to pick names such as 'alp CMa'. Default: ['HIP','HD','HR'].
    Output:
        - a dictionary with the most interesting simbad keywords and the original 
            RA,DEC coordinates from the pointing position.
    """
    search_radius = 10*u.arcsec # we search in a 10arcsec circle.
    search_radius_alt = 220*u.arcsec # in case nothing is found, we enlarge the search
        # we use 210 arcsec because Barnard star (higher PM star moves by 10arcsec/yr --> 220 arcsec in 22yrs)

    if isinstance(name,str):
        n_obj=1
        name=np.array([name])    
        if coords.ndim>0: # if coords is an array of SkyCoord we only take the 2st element to avoid issues with arrays.
            coords = coords[0]
    elif type(name)==type(None):
        n_obj=0
    else:
        n_obj=len(name)
        name=np.array(name).astype(str)    

    # The output of the function is simbad_dico.
    # We first populate it with the initial values to pass 
    simbad_dico = {}
    simbad_dico['RA'] = coords.ra.to_string(unit=u.hourangle,sep=' ')
    simbad_dico['DEC'] = coords.dec.to_string(unit=u.degree,sep=' ')
    simbad_dico['DATE'] = date.iso
    if type(metadata) is dict:
        for key,val in metadata.items():
            simbad_dico[key] = val           

    if name is not None:
        planet_ends, planet_nums = [], np.concatenate((np.arange(0,10).astype('str'),[' ']))
        for j in ['a','b','c','d','e','f','g','h']: planet_ends.extend([i+j for i in planet_nums])
        
        for i in range(n_obj):
            name_i = name[i]
            coord = coords if n_obj==1 else coords[i]
            # here we can handle special cases where the object name is 47 Tuc for instance
            if np.logical_and('47' in name_i,'tuc' in name_i.lower()):
                name_i = 'Gaia EDR3 4689637789368594944'
            elif np.logical_and('3603' in name_i,'ngc' in name_i.lower()):
                name_i ='HD 97950B'
                # to be checked ! I'm not sure this is the AO star...
                print('NGC 3603 case not implemented yet')
            elif np.logical_and('6380' in name_i,'ngc' in name_i.lower()):
                name_i = 'Gaia EDR3 5961801153907816832'
            elif np.logical_and('theta' in name_i.lower(),'ori' in name_i.lower()):
                # I still have to find the coordinate of theta Ori B1 which is the other astrometric calibrator often used. 
                name_i = 'tet01 Ori B'
            if name_i.startswith('NAME '):
                name_i=name_i[5:]
            if name_i.startswith('Vstar '):
                name_i=name_i[6:]
            if name_i.endswith(' System'):
                name_i=name_i[:-7]
            if name_i[-2:] in planet_ends:
                wrong_name = ''+name_i
                name_i=name_i[:-1]
                if verbose:
                    print('Warning! Wrong identifier detected: a planet ({0}) was used instead of its star ({1}). Fixing...'.format(wrong_name,name_i))

            # then we try to resolve the name of the object directly.
            try:
                object_coordinate = SkyCoord.from_name(name_i.strip())
                separation = object_coordinate.separation(coord)
                if verbose:
                    print('Object - Pointing separation is {0:.2f}'.format(separation))
                if separation < search_radius_alt:
                    if verbose:
                        print('The object found is likely the target')
                    name_i = name_i.strip()
                else:
                    if verbose:
                        print('The object found is likely not the target.')
                    name_i = None
            except name_resolve.NameResolveError as e:
                if verbose:
                    print('Object {} not recognized'.format(name_i.strip()))
                    print(e)
                name_i = None
            name[i]=name_i

        # at this point we have done our best to have a valid name recognized by 
        # Simbad. If this is the case, then name is a valid string. Otherwise, name is None.

    customSimbad = Simbad()

    binary_ends, binary_nums = [], np.concatenate((np.arange(0,10).astype('str'),[' ']))
    for j in ['A','B','C','D','E','F']: binary_ends.extend([i+j for i in binary_nums])
    bin_flag = np.zeros(n_obj,dtype=str)
    for j in range(n_obj):
        if name[j][-2:] in binary_ends:
            bin_flag[j]=name[j][-1]

    if name is not None:

        if n_obj==1:

            customSimbad.add_votable_fields('typed_id','ids','flux(U)','flux(B)','flux(V)','flux(R)',\
                                            'flux(I)','flux(G)','flux(J)','flux(H)',\
                                            'flux(K)','id(HD)','sp','otype','otype(V)','otype(3)',\
                                           'propermotions','ra(2;A;ICRS;J2000;2000)',\
                                         'dec(2;D;ICRS;J2000;2000)',\
                                         'ra(2;A;FK5;J{0:.3f};2000)'.format(date.jyear),\
                                         'dec(2;D;FK5;J{0:.3f};2000)'.format(date.jyear))
            search = customSimbad.query_objects(name)
        else:
            customSimbad.add_votable_fields('typed_id','ids','flux(U)','flux(B)','flux(V)','flux(R)',\
                                            'flux(I)','flux(G)','flux(J)','flux(H)',\
                                            'flux(K)','id(HD)','sp','otype','otype(V)','otype(3)',\
                                           'propermotions','ra(2;A;ICRS;J2000;2000)',\
                                         'dec(2;D;ICRS;J2000;2000)')           
            search = customSimbad.query_objects(name)
            search = fix_binaries(search,name,bin_flag,customSimbad)

            RA1, DEC1 = [], []
            for i in range(n_obj):
                if name[i]!='None':
                    dt=(Time(date[i])-Time('J2000.0')).jd*u.day.to(u.yr)
                    RA0, DEC0, PMRA0, PMDEC0 = search['RA_2_A_ICRS_J2000_2000'][i], search['DEC_2_D_ICRS_J2000_2000'][i],\
                    search['PMRA'][i], search['PMDEC'][i]
                    coo=SkyCoord(ra=RA0,dec=DEC0,unit=(u.hourangle,u.deg),frame='icrs')
                    coo1=SkyCoord(ra=coo.ra+PMRA0*u.mas/u.yr*dt*u.yr,dec=coo.dec+PMDEC0*u.mas/u.yr*dt*u.yr,frame='icrs')
                    coo1_string = (coo1.transform_to(FK5)).to_string('hmsdms').split(' ')
                    ra1_string = coo1_string[0].replace('h',' ').replace('m',' ').replace('s','').split('.')
                    dec1_string = coo1_string[1].replace('d',' ').replace('m',' ').replace('s','').split('.')
                    RA1.append(ra1_string[0]+str(round(float('0.'+ra1_string[1]), 4))[1:])
                    DEC1.append(dec1_string[0]+str(round(float('0.'+dec1_string[1]), 4))[1:])
                else:
                    RA1.append('')
                    DEC1.append('')
                    
            search['RA_2_A_FK5_obstime']=RA1
            search['DEC_2_A_FK5_obstime']=DEC1
                

        best_names=get_best_id(search,pref_order)
        search['BEST_NAME'] = best_names 
        search['BIN_FLAG'] = bin_flag
        del search['IDS']
        
        filter_cols = ['FLUX_G', 'FLUX_V', 'FLUX_R','FLUX_U', 'FLUX_B','FLUX_I','FLUX_J', 'FLUX_H']
        nb_stars, i = 0, 0
        if n_obj==1:
            if type(search)!=type(None):
                while (nb_stars==0) & (i<len(filter_cols)): # if the star is fainter than that, it's likely not the one we are looking for
                    validSearch = search[search[filter_cols[i]]<limit_G_mag]
                    nb_stars = len(validSearch)
                    i+=1
                # sometimes no photometry is available on Simbad(). Depending on 
                # the truth value of 'force_cm', one can accept the cross-matched object or reject it 
                if (nb_stars==0) & (force_cm):
                    if verbose:
                        print('No photometry available for this star, check it carefully.')
                    nb_stars=len(search)
                    validSearch=search
            else: nb_stars=0
        
            if nb_stars == 1:
                simbad_dico = populate_simbad_dico(validSearch,0,simbad_dico)

                # we add the distance between pointing and current position in the dictionary
                simbad_dico  = add_separation_between_pointing_current_position(coords,simbad_dico)
                return simbad_dico

            else:
                if verbose:
                    print('Something went wrong, there are {0:d} valid stars '.format(nb_stars))
                return None
            
        else:
            if type(search)!=type(None):
                mask = (search[filter_cols[i]]<-30).filled(False)
                while (nb_stars<len(search)) & (i<len(filter_cols)): # if the star is fainter than that, it's likely not the one we are looking for
                    mask += (search[filter_cols[i]]<limit_G_mag).filled(False)
                    nb_stars = np.sum(mask)
                    i+=1
                if (nb_stars<len(search)) & (force_cm):
                    if verbose:
                        print('No photometry available for {0} stars, be careful.'.format(len(search)-nb_stars))
                    nb_stars=len(search)
            else: nb_stars=0

            if nb_stars == len(search):
                simbad_dico = populate_simbad_dico(search,None,simbad_dico)
                # we add the distance between pointing and current position in the dictionary
                simbad_dico  = add_separation_between_pointing_current_position(coords,simbad_dico)                
                return simbad_dico
            else:
                if verbose:
                    print('No photometry was found for {0} stars, so they were rejected.'.format(len(search)-nb_stars))
                search = Table(search, masked=True)
                for col in search.columns:
                    if col=='TYPED_ID': continue
                    search[col].mask=~mask
                simbad_dico = populate_simbad_dico(search,None,simbad_dico)
                simbad_dico  = add_separation_between_pointing_current_position(coords,simbad_dico)
                return simbad_dico

    else: # in this case no name is provided
    
        # First we do a cone search around the coordinates
        search = customSimbad.query_region(coords,radius=search_radius)
        if search is not None:
            validSearch = search[search['FLUX_G']<limit_G_mag]
            nb_stars = len(validSearch)                
            if nb_stars==0:
                search = None
        if search is  None:
            # If the cone search failed and no name is provided we cannot do anything more
            if verbose:
                print('No star identified for the RA/DEC pointing. Enlarging the search to {0:.0f} arcsec'.format(search_radius_alt.value))
            search = customSimbad.query_region(coords,radius=search_radius_alt)
            if search is None:
                if verbose:
                    print('No star identified for the RA/DEC pointing. Stopping the search.')
                return simbad_dico
            else:
                validSearch = search[search['FLUX_G']<limit_G_mag]
                nb_stars = len(validSearch)                
                        
        if nb_stars==0:
            print('No star identified for the RA/DEC pointing. Stopping the search.')
            return simbad_dico
        elif nb_stars>0:
            if nb_stars ==1:
                i_min=0
                if verbose:
                    print('One star found: {0:s} with G={1:.1f}'.format(\
                        validSearch['MAIN_ID'][i_min],validSearch['FLUX_G'][i_min]))
            else:
                print('{0:d} stars identified within {1:.0f} or {2:.0f} arcsec'.format(nb_stars,search_radius.value,search_radius_alt.value)) 
                print('Target not resolved or not in the list. Selecting the closest star.')
                sep_list = []
                for key in validSearch.keys():
                    if key.startswith('RA_2_A_FK5_'):
                        key_ra_current_epoch = key
                    elif key.startswith('DEC_2_D_FK5_'):
                        key_dec_current_epoch = key
                for i in range(nb_stars):
                    ra_i = validSearch[key_ra_current_epoch][i]
                    dec_i = validSearch[key_dec_current_epoch][i]
                    coord_str = ' '.join([ra_i,dec_i])
                    coords_i = SkyCoord(coord_str,frame=FK5,unit=(u.hourangle,u.deg))
                    sep_list.append(coords.separation(coords_i).to(u.arcsec).value)
                i_min = np.argmin(sep_list)
                min_sep = np.min(sep_list)
                print('The closest star is: {0:s} with G={1:.1f} at {2:.2f} arcsec'.format(\
                  validSearch['MAIN_ID'][i_min],validSearch['FLUX_G'][i_min],min_sep))
        simbad_dico = populate_simbad_dico(validSearch,i_min,simbad_dico)
        simbad_dico = add_separation_between_pointing_current_position(coords,simbad_dico,verbose=verbose)
        if verbose:
            print_dico_results(simbad_dico)        
        return simbad_dico

def populate_simbad_dico(simbad_search_list,i,simbad_dico):
    """
    Method not supposed to be used outside the query_simbad method
    Given the result of a simbad query (list of simbad objects), and the index of 
    the object to pick, creates a dictionary with the entries needed.
    """
    
    if i==None:
        for key in simbad_search_list.keys():
            if key in ['MAIN_ID','BEST_NAME','SP_TYPE','ID_HD','OTYPE','OTYPE_V','OTYPE_3']: #strings
                simbad_dico['simbad_'+key] = np.array(simbad_search_list[key])
            elif key in ['FLUX_G', 'FLUX_J', 'FLUX_H', 'FLUX_K','PMDEC','PMRA']: #floats
                simbad_dico['simbad_'+key] = np.array(simbad_search_list[key],dtype=float)
            elif key.startswith('RA_2_A_FK5_'): 
                simbad_dico['simbad_RA_current'] = simbad_search_list[key]
            elif key.startswith('DEC_2_D_FK5_'): 
                simbad_dico['simbad_DEC_current'] = simbad_search_list[key]
            elif key=='RA':
                simbad_dico['simbad_RA_ICRS'] = simbad_search_list[key]
            elif key=='DEC':
                simbad_dico['simbad_DEC_ICRS'] = simbad_search_list[key]
    else:
        for key in simbad_search_list.keys():
            if key in ['MAIN_ID','BEST_NAME','SP_TYPE','ID_HD','OTYPE','OTYPE_V','OTYPE_3']: #strings
                simbad_dico['simbad_'+key] = simbad_search_list[key][i]
            elif key in ['FLUX_G', 'FLUX_J', 'FLUX_H', 'FLUX_K','PMDEC','PMRA']: #floats
                simbad_dico['simbad_'+key] = float(simbad_search_list[key][i])
            elif key.startswith('RA_2_A_FK5_'): 
                simbad_dico['simbad_RA_current'] = simbad_search_list[key][i]      
            elif key.startswith('DEC_2_D_FK5_'): 
                simbad_dico['simbad_DEC_current'] = simbad_search_list[key][i]
            elif key=='RA':
                simbad_dico['simbad_RA_ICRS'] = simbad_search_list[key][i]
            elif key=='DEC':
                simbad_dico['simbad_DEC_ICRS'] = simbad_search_list[key][i]
                
    return simbad_dico

def add_separation_between_pointing_current_position(coords,simbad_dico,verbose=False):
    """
    Input: 
        - coords: a SkyCoord object. For instance, if we extract the keywords 
            of the fits files, we should use
            coords = SkyCoord(header['RA']*u.degree,header['DEC']*u.degree)
            SkyCoord('03h32m55.84496s -09d27m2.7312s', ICRS)
        - simbad_dico is a dictionary containing the keys 
            ['simbad_MAIN_ID',
             'simbad_SP_TYPE',
             'simbad_ID_HD',
             'simbad_OTYPE',
             'simbad_OTYPE_V',
             'simbad_OTYPE_3',
             'simbad_FLUX_G',
             'simbad_FLUX_J',
             'simbad_FLUX_H',
             'simbad_FLUX_K',
             'simbad_PMDEC',
             'simbad_PMRA',
             'simbad_simbad_RA_current',
             'simbad_simbad_DEC_current',
             'simbad_simbad_RA_ICRS',
             'simbad_simbad_DEC_ICRS']
    The function adds the keys simbad_separation_RADEC_ICRSJ2000 and simbad_separation_RADEC_current
    corresponding to the distance between pointing and ICRS and current coordinates    
    It returns the updated dictionary
    """
    try:
        if 'simbad_RA_ICRS' in simbad_dico.keys() and 'simbad_DEC_ICRS' in simbad_dico.keys():
            coords_ICRS_str = ' '.join([simbad_dico['simbad_RA_ICRS'],simbad_dico['simbad_DEC_ICRS']])
            coords_ICRS = SkyCoord(coords_ICRS_str,frame=ICRS,unit=(u.hourangle,u.deg))
            sep_pointing_ICRS = coords.separation(coords_ICRS).to(u.arcsec).value
            simbad_dico['simbad_separation_RADEC_ICRSJ2000'] = sep_pointing_ICRS
        # if we found a star, we add the distance between Simbad current coordinates and pointing
        if 'simbad_RA_current' in simbad_dico.keys() and 'simbad_DEC_current' in simbad_dico.keys():
            coords_current_str = ' '.join([simbad_dico['simbad_RA_current'],simbad_dico['simbad_DEC_current']])
            coords_current = SkyCoord(coords_current_str,frame=ICRS,unit=(u.hourangle,u.deg))
            sep_pointing_current = coords.separation(coords_current).to(u.arcsec).value
            simbad_dico['simbad_separation_RADEC_current']=sep_pointing_current
            if verbose:
                print('Distance between the current star position and pointing position: {0:.1f}arcsec'.format(simbad_dico['simbad_separation_RADEC_current']))
    except TypeError:
        n_obj=len(simbad_dico['simbad_MAIN_ID'])
        if 'simbad_RA_ICRS' in simbad_dico.keys() and 'simbad_DEC_ICRS' in simbad_dico.keys():
            sep_pointing_ICRS=[]
            for i in range(n_obj):
                if simbad_dico['simbad_RA_ICRS'][i]!='':
                    coords_ICRS_str = ' '.join([simbad_dico['simbad_RA_ICRS'][i],simbad_dico['simbad_DEC_ICRS'][i]])
                    coords_ICRS = SkyCoord(coords_ICRS_str,frame=ICRS,unit=(u.hourangle,u.deg))
                    sep_pointing_ICRS.append(coords[i].separation(coords_ICRS).to(u.arcsec).value)
                else: sep_pointing_ICRS.append(np.nan)
            simbad_dico['simbad_separation_RADEC_ICRSJ2000'] = sep_pointing_ICRS
        # if we found a star, we add the distance between Simbad current coordinates and pointing
        if 'simbad_RA_current' in simbad_dico.keys() and 'simbad_DEC_current' in simbad_dico.keys():
            sep_pointing_current=[]
            for i in range(n_obj):
                if simbad_dico['simbad_RA_current'][i]!='':
                    coords_current_str = ' '.join([simbad_dico['simbad_RA_current'][i],simbad_dico['simbad_DEC_current'][i]])
                    coords_current = SkyCoord(coords_current_str,frame=ICRS,unit=(u.hourangle,u.deg))
                    sep_pointing_current.append(coords[i].separation(coords_current).to(u.arcsec).value)
                else: sep_pointing_current.append(np.nan)
            simbad_dico['simbad_separation_RADEC_current']=sep_pointing_current
            if verbose:
                for i in range(n_obj):
                    print('Distance between the current star {1} positions and pointing positions: {0:.1f}arcsec'.format(simbad_dico['simbad_BEST_NAME'][i],simbad_dico['simbad_separation_RADEC_current'][i]))

            
    return simbad_dico

def is_moving_object(header):
    """
    Parameters
    ----------
    header : dictionary
        Header of the fits file.

    Returns
    -------
    bool
        True if differential tracking is used by the telescope, indicating a
        moving target therefore not listed in Simbad.
    """
    if 'ESO TEL TRAK STATUS' in header.keys():
        if 'DIFFERENTIAL' in header['HIERARCH ESO TEL TRAK STATUS']:
            return True
    return False

def print_dico_results(dico):
    """
    Ancilliary function that prints on screen the information contained in a dictionary
    """
    if dico is not None:
        for index,key in enumerate(dico):
            print(key,dico[key])
    return

if __name__ == "__main__":
    """
    This is just an example of how the script can be used
    """

    path_data = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')
    
    print('\n\n','-'*20)
    ra = 10.*u.degree
    dec = -24*u.degree
    name='eps Eri'
    testCoord = SkyCoord(ra,dec)
    date = Time('2017-01-01T02:00:00.0')
    print("Let's query a random coordinates ra={0:s} dec={1:s} with the name {2:s} and see what's happening\n".format(testCoord.ra,testCoord.dec,name))
    test=query_simbad(date,testCoord,name=name,limit_G_mag=15,verbose=True)
    # for index,key in enumerate(test):
    #     print(key,test[key])
    
    print('\n\n','-'*20)
    # the following example uses real data from the observations of Barnard's star in 2019. 
    # it shows that the pointing coordinate (displayed as RA and DEC in the header) correspond
    # to the current coordinates at the time of observations and are not corrected by proper motion
    testCoord = SkyCoord('17:57:47.3 +04:44:59.0', frame=ICRS, unit=(u.hourangle, u.deg))
    date = Time('2019-08-07T01:16:53.3630')
    print("Let's query a star at ra={0:s} dec={1:s} observed on {2:s}\n".format(testCoord.ra,testCoord.dec,str(date)))
    test=query_simbad(date,testCoord,name='Barnard',limit_G_mag=15)    

    print('\n\n','-'*20)
    h = fits.getheader(os.path.join(path_data,'SPHER.2019-04-01T03-39-17.958IRD_SCIENCE_DBI_RAW.fits'))
    print("Let's query a target from a real SPHERE header\n")
    test = query_simbad_from_header(h)
        
    print('\n\n','-'*20)
    h = fits.getheader(os.path.join(path_data,'SPHER.2019-02-25T03-55-45.738ZPL_SCIENCE_IMAGING_RAW.fits'))
    print("Let's query a target from a real SPHERE header (a moving target in this case) \n")
    test = query_simbad_from_header(h)
    
    print('\n\n','-'*20)    
    ra = 6.01*u.degree
    dec = -72.09*u.degree
    name='47 Tuc'
    testCoord = SkyCoord(ra,dec)
    date = Time('2017-01-01T02:00:00.0')
    print("Let's query 47 Tuc at ra={0:s} dec={1:s} with the name {2:s} and see what's happening\n".format(testCoord.ra,testCoord.dec,name))
    test=query_simbad(date,testCoord,name=name,limit_G_mag=15,verbose=True)
    
