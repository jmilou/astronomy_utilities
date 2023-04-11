#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:22:00 2022

@author: millij
modified by Johan M with verbose option to avoid printing
modified by Vito S to allow vectorized queries; to resolve some targets with bad spelling,
wrong object name, not found in Gaia, or not present in Simbad; 
to enable a preferred order of naming conventions; to handle binaries
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
from astropy.table import Table, vstack
import re
from astroquery.vizier import Vizier


FILTER_COLUMNS = ['FLUX_G', 'FLUX_V', 'FLUX_R','FLUX_U', 'FLUX_B','FLUX_I','FLUX_J', 'FLUX_H']
SEARCH_RADIUS = 10*u.arcsec # first search performed inside a 10 arcsec circle
SEARCH_RADIUS_ALT = 220*u.arcsec # in case nothing is found, we enlarge the search
# we use 210 arcsec because Barnard star (higher PM star moves by 10arcsec/yr --> 220 arcsec in 22yrs)

# binary and planet names in Simbad end with lower and upper case letters, respectively
PLANET_ENDS, planet_nums = [], np.concatenate((np.arange(0,10).astype('str'),[' ']))
for j in ['a','b','c','d','e','f','g','h']: PLANET_ENDS.extend([i+j for i in planet_nums])
BINARY_ENDS, binary_nums = [], np.concatenate((np.arange(0,10).astype('str'),[' ']))
for j in ['A','B','C','D','E','F']: BINARY_ENDS.extend([i+j for i in binary_nums])


def _find_brightest_star(search):
    """
    Method not supposed to be used outside the query_simbad method
    Given a result table as returned from Simbad, identifies the row index
    corresponding to the brightest star
    """
    
    i=0
    checked = np.zeros(len(search),dtype=bool)
    to_check = np.ones(len(search),dtype=bool)
    min_mags, min_ind = [], []
    while i<len(FILTER_COLUMNS):
        phot_vector = np.array(search[FILTER_COLUMNS[i]].filled(np.nan))
        phot_vector[~to_check] = np.nan
        no_phot = np.isnan(phot_vector)
        n_bad, n_good = np.sum(no_phot), np.sum(~no_phot)
        if n_good>0:
            ind = np.argmin(phot_vector[~no_phot])
            checked[~no_phot] = True
            to_check[~no_phot] = False
            true_ind = np.where(~no_phot)[0][ind]
            to_check[true_ind] = True
            min_mags.append(phot_vector[true_ind])
            min_ind.append(true_ind)
        if np.sum(checked)==len(search): break
        
        i+=1
    
    min_mags, min_ind = np.array(min_mags), np.array(min_ind)
    if len(min_ind)==0: return -1
    elif len(np.unique(min_ind))==1:
        return min_ind[0]
    else:
        i_min = np.argmin(min_mags)
        return min_ind[i_min]

def _remove_duplicate_entries(search):
    """
    Method not supposed to be used outside the query_simbad method
    Given a result table as returned from Simbad, removes entries for binary systems
    when individual entries for A components are present
    e.g., if both HD 126838 and HD 126838A are returned by the query, the former is removed
    """
    
    if search is None: return search
    elif len(search)<2: return search
    
    ids = np.array(search['MAIN_ID'].filled(''))
    is_there_A = [i[-2:] in BINARY_ENDS[0:11] for i in ids] # name ending in '1A', ..., '9A' or ' A'
    if np.sum(is_there_A)==0:
        return search
    else:
        keep = list(np.arange(0,len(search)))
        w,=np.where(is_there_A)
        for i in w: 
            ids[i] = ids[i][:-1].rstrip()
            w_eq, = np.where(ids==ids[i])
            if len(w_eq)>1:
                del_ind = int(np.setdiff1d(w_eq,i))
                del keep[del_ind]
        return search[keep]

def _get_best_id(simbad_table,pref_order):
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
                        best_ids.append(" ".join(match.group(0).split()))
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
                    if (match1 is None) & (match2 is None) & (match is not None):
                        best_ids.append(" ".join(id_list[j].split()))
                        break_cycle = True
                        break
                    j+=1
                if break_cycle: break
                elif j==len(id_list): pref_i+=1
            else:
                ww,=np.where(np.char.find(id_list,pref_order[pref_i])!=-1)
                if len(ww)>=1:
                    best_ids.append(" ".join(id_list[ww[0]].split()))
                    break
                else:
                    pref_i+=1
        if pref_i==len(pref_order): best_ids.append(" ".join(default_names[i].split()))

    return best_ids

def _fix_binaries(table,objects,bin_flag,SimbadQuery):
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
            if results_table is not None:
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

def query_simbad_from_header(header,**kwargs):
    """
    Function similar to query_simbad, but using the date, coord and name extracted 
    from one or more header(s)
    All optional inputs as in query_simbad() are accepted
    Input:
        - header: the header of a SPHERE file, or a list of headers if more than one star is to be queried
    """
    
    if isinstance(header,list) == False: header = [header]
    
    is_moving = np.zeros(len(header),dtype=bool)
    date, ra, dec, name = [], [], [], []
    for i,h in enumerate(header):
        try:
            if is_moving_object(h):
                #print('The object is a moving target and no information can be retrieved from Simbad')
                is_moving[i] = True
        except KeyError: pass
        date.append(h['DATE-OBS'])
        ra.append(h['RA'])
        dec.append(h['DEC'])
        name.append(h['OBJECT'])
        
    date = Time(date)
    coords = SkyCoord(ra*u.degree,dec*u.degree,frame=ICRS)
    
    if np.sum(is_moving)==len(header):
        print('All queried objects are moving targets; no information can be retrieved from Simbad. Returning None.')
        return None
    else:
        return query_simbad(date,coords,name=name,**kwargs)

def is_name_null(name):
    """
    Returns True if the input is 1) an empty string, 2) a 'nan' string or 3) a nan float, False otherwise
    """
    if isinstance(name,str):
        if (name.strip()=='') | (name.lower()=='nan'):
            return True
        else: return False
    elif isinstance(name,float):
        if np.isnan(name):
            return True
        else: return False
    else: return False

def _query_simbad_from_coords(date,coords,force_verbose=None,**kwargs):
    """
    Method not supposed to be used outside the query_simbad method
    Starting only from observing times and coordinates, tries to resolve the most likely object
    When more than one object is found, either the closest to input coords or the brightest in the queried FoV
    is selected, depending on the value of the keyword 'select'
    """
    
    verbose, limit_G_mag, select = kwargs['verbose'], kwargs['limit_G_mag'], kwargs['select']
    enlarge_query = kwargs['enlarge_query'] if 'enlarge_query' in kwargs else False
    if force_verbose is not None: verbose = force_verbose

    # we use 210 arcsec because Barnard star (highest PM star) moves by 10 arcsec/yr --> 220 arcsec in 22 yrs

    customSimbad = Simbad()
    customSimbad.TIMEOUT = 600    
    customSimbad.add_votable_fields('typed_id','ids','flux(U)','flux(B)','flux(V)','flux(R)',\
                                    'flux(I)','flux(G)','flux(J)','flux(H)',\
                                    'flux(K)','id(HD)','sp','otype','otype(V)','otype(3)',\
                                   'parallax','propermotions','ra(2;A;ICRS;J2000;2000)',\
                                 'dec(2;D;ICRS;J2000;2000)',\
                                 'ra(2;A;FK5;J{0:.3f};2000)'.format(date.jyear),\
                                 'dec(2;D;FK5;J{0:.3f};2000)'.format(date.jyear))

    nb_stars, i = 0, 0
    
    # First we do a cone search around the coordinates (radius = SEARCH_RADIUS)
    search = customSimbad.query_region(coords,radius=SEARCH_RADIUS,cache=False)
    if search is not None:
        select_stars = np.zeros(len(search),dtype=bool)
        while i<len(FILTER_COLUMNS): # to cope with stars not found in Gaia, we also look for UBV and 2MASS photometry
            select_stars[search[FILTER_COLUMNS[i]]<limit_G_mag] = True # if the star is fainter than that, it's likely not the one we are looking for
            i+=1
        validSearch = search[select_stars]
        nb_stars = len(validSearch)
    # If the first search failed, we try to enlarge the FOV (radius = SEARCH_RADIUS_ALT)
#    if (search is None) & enlarge_query:
    if search is None:
        if verbose:
            print('  No star identified for the RA/DEC pointing. Enlarging the search to {0:.0f} arcsec'.format(SEARCH_RADIUS_ALT.value))
        search = customSimbad.query_region(coords,radius=SEARCH_RADIUS_ALT,cache=False)
        if search is None:
            if verbose:
                print('  No star identified for the RA/DEC pointing. Stopping the search.')
            return None
        else:
            select_stars = np.zeros(len(search),dtype=bool)
            while i<len(FILTER_COLUMNS): # to cope with stars not found in Gaia, we also look for UBV and 2MASS photometry
                select_stars[search[FILTER_COLUMNS[i]]<limit_G_mag] = True # if the star is fainter than that, it's likely not the one we are looking for
                i+=1
        validSearch = search[select_stars]
        nb_stars = len(validSearch)
#        final_radius = SEARCH_RADIUS_ALT
#    else: final_radius = SEARCH_RADIUS
    final_radius = SEARCH_RADIUS_ALT
        

    if nb_stars==0: # If no star is found, None is returned
        if verbose: print('  No star identified for the RA/DEC pointing. Stopping the search.')
        best_match_ID = None
    elif nb_stars>0:
        validSearch = _remove_duplicate_entries(validSearch)
        nb_stars = len(validSearch)
        if nb_stars == 1: # If just one star is found, we are fine with it
            i_min=0
            if verbose:
                j = 0
                while j<len(FILTER_COLUMNS):
                    if np.isnan(validSearch[FILTER_COLUMNS[j]][i_min]) == False: break
                    j+=1
                print('   Selecting the only star found: {0:s} with {1}={2:.1f}'.format(\
                    validSearch['MAIN_ID'][i_min],FILTER_COLUMNS[j].replace('FLUX_',''),validSearch[FILTER_COLUMNS[j]][i_min]))
            best_match_ID=_get_best_id(validSearch,kwargs['pref_order'])[0]
        else: # If more than one star is found, only one is chosen according to the criterion indicated by the 'select' keyword
            if verbose:
                print('   {0:d} stars identified within {1:.0f} arcsec'.format(nb_stars,final_radius.value))
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
            min_sep = np.min(sep_list)
            if enlarge_query==False:
                if min_sep>SEARCH_RADIUS.value: 
                    if verbose:
                        print('   None of them has a separation at the observing epoch < {0}. Returning None.'.format(SEARCH_RADIUS))
                    return None
            
            if select=='closest': i_min = np.argmin(sep_list)
            elif select=='brightest':
                brightest_failed = False
                if enlarge_query:                
                    i_min = _find_brightest_star(validSearch)
                else:
                    w,=np.where(sep_list<SEARCH_RADIUS.value)
                    i_min = w[_find_brightest_star(validSearch[sep_list<SEARCH_RADIUS.value])]
                if i_min==-1:
                    brightest_failed = True
                    i_min = np.argmin(sep_list)
                    
            if verbose:
                j = 0
                while j<len(FILTER_COLUMNS):
                    if np.isnan(validSearch[FILTER_COLUMNS[j]][i_min]) == False: break
                    j+=1
                if select=='closest':
                    print('   Selecting the closest star: {0:s} with {1}={2:.1f} at {3:.2f} arcsec'.format(\
                      validSearch['MAIN_ID'][i_min],FILTER_COLUMNS[j].replace('FLUX_',''),validSearch[FILTER_COLUMNS[j]][i_min],min_sep))
                elif select=='brightest':
                    if brightest_failed:
                        print('   Selected criterion "brightest" did not work due to lacking magnitudes in Simbad. Reverting to "closest..."')
                        print('   Selecting the closest star: {0:s} with {1}={2:.1f} at {3:.2f} arcsec'.format(\
                          validSearch['MAIN_ID'][i_min],FILTER_COLUMNS[j].replace('FLUX_',''),validSearch[FILTER_COLUMNS[j]][i_min],min_sep))
                    else:
                        print('   Selecting the brightest star: {0:s} with {1}={2:.1f} at {3:.2f} arcsec'.format(\
                          validSearch['MAIN_ID'][i_min],FILTER_COLUMNS[j].replace('FLUX_',''),validSearch[FILTER_COLUMNS[j]][i_min],min_sep))
            best_match_ID=_get_best_id(validSearch[[i_min]],kwargs['pref_order'])[0]
                
        return best_match_ID

def _vizier_resolver(name,coords,search,index):
    """
    Method not supposed to be used outside the query_simbad method
    Not every star with V<15 is in Simbad: this method tries to collect information for objects
    resolved by the CDS but not in Simbad and to add them to the Simbad dataset. In particular,
    it queries 2MASS and Gaia DR3 catalogs. From the former it retrieves name, J2000 coordinates and JHK mags;
    from the latter, J2016 and J2000 coordinates (if not already collected), G mags, parallaxes
    and proper motions.
    """
    
    if search is None: return None
    
    in_2mass = True
    try:
        res_2mass = Vizier.query_object(name,catalog=['II/246/out'],radius=SEARCH_RADIUS)[0]
    except IndexError:
        in_2mass = False
    in_gaia = True
    try:
        res_gaia = Vizier.query_object(name,catalog=['I/355/gaiadr3'],radius=SEARCH_RADIUS)[0]
    except IndexError:
        in_gaia = False
    
    if (in_2mass == False) & (in_gaia == False): return search
    
    coo_added = False
    saved_names = name+''
    if in_2mass:
    
        dict_2mass = {'RA':'RAJ2000','DEC':'DEJ2000','FLUX_J':'Jmag','FLUX_H':'Hmag','FLUX_K':'Kmag',
                      'RA_2_A_ICRS_J2000_2000':'RAJ2000','DEC_2_D_ICRS_J2000_2000':'DEJ2000'}
        res_2mass_ids = np.array(['2MASS J'+i for i in res_2mass['_2MASS']])
        res_2mass_coo = SkyCoord(ra=res_2mass['RAJ2000'],dec=res_2mass['DEJ2000'],frame=ICRS)
        
        if name.startswith('2MASS'):
            w, = np.where((res_2mass_coo.separation(coords)<SEARCH_RADIUS) & (res_2mass_ids == name))
        else:
            sep_list = res_2mass_coo.separation(coords)
            w = np.nanargmin(sep_list)
            if sep_list[w] > SEARCH_RADIUS: 
                w = []
            else:
                saved_names += ('|'+res_2mass_ids[w])
        
        if hasattr(w,'__len__')==False: w = [w]
        if len(w)>0:
            if len(w)>1: w=w[0]
            coord_2mass = SkyCoord(ra=res_2mass['RAJ2000'][w],dec=res_2mass['DEJ2000'][w],frame=ICRS)
            coo_added = True

            for col in dict_2mass.keys():
                if col.startswith('RA'):
                    search[col][index] = coord_2mass[0].ra.to_string(unit=u.hour, sep=' ')
                elif col.startswith('DEC'):
                    search[col][index] = coord_2mass[0].dec.to_string(unit=u.deg, sep=' ')
                else:
                    search[col][index] = res_2mass[dict_2mass[col]][w][0]
            search['OTYPE_V'][index] = 'Star'
            search['OTYPE'][index] = 'Star'
            search['OTYPE_3'][index] = '*'
            search['MAIN_ID'][index] = res_2mass_ids[w][0]
    
    if in_gaia:
        dict_gaia = {'PLX_VALUE':'Plx','PLX_ERROR':'e_Plx','PMRA':'pmRA','PMDEC':'pmDE','FLUX_G':'Gmag'}
        
        res_gaia_coo = SkyCoord(ra=res_gaia['RA_ICRS'],dec=res_gaia['DE_ICRS'],frame=ICRS)

        if name.startswith('2MASS'):
            res_gaia_ids = np.array(['2MASS J'+i for i in res_gaia['_2MASS']])
            w, = np.where(res_gaia_ids == name)
            if len(w)>0:
                saved_names += ('|Gaia DR3 '+str(res_gaia['Source'][w][0]))
        elif name.startswith('Gaia DR3'):
            res_gaia_ids = np.array(['Gaia DR3'+i for i in res_gaia['Source']])
            w, = np.where(res_gaia_ids == name)
        else:
            sep_list = res_gaia_coo.separation(coords)
            w = np.nanargmin(sep_list)
            if sep_list[w] > SEARCH_RADIUS: 
                w = []
            else:
                saved_names += ('|Gaia DR3 '+str(res_gaia['Source'][w]))
                
        if hasattr(w,'__len__')==False: w = [w]
        if len(w)>0:
            if len(w)>1: w=w[0]
            
            if coo_added==False:
                coord_gaia = SkyCoord(ra=res_gaia['RA_ICRS'][w],dec=res_gaia['DE_ICRS'][w],frame=ICRS)
                PMRA0, PMDEC0 = res_gaia['pmRA'].value[w], res_gaia['pmDE'].value[w]
                coord_gaia_J2000 = SkyCoord(ra=coord_gaia.ra-PMRA0*u.mas*16,dec=coord_gaia.dec-PMDEC0*u.mas*16,frame='icrs')
                
                search['RA'][index] = coord_gaia[0].ra.to_string(unit=u.hour, sep=' ')
                search['RA_2_A_ICRS_J2000_2000'][index] = coord_gaia_J2000[0].ra.to_string(unit=u.hour, sep=' ')
                search['DEC'][index] = coord_gaia[0].dec.to_string(unit=u.deg, sep=' ')
                search['DEC_2_D_ICRS_J2000_2000'][index] = coord_gaia_J2000[0].dec.to_string(unit=u.deg, sep=' ')
                
                search['MAIN_ID'][index] = 'Gaia DR3 '+str(res_gaia['Source'][w][0])
            
            for col in dict_gaia.keys():
                search[col][index] = res_gaia[dict_gaia[col]][w][0]
            search['OTYPE_V'][index] = 'Star'
            search['OTYPE'][index] = 'Star'
            search['OTYPE_3'][index] = '*'
            if coo_added == False:
                
                if col.startswith('RA'):
                    search[col][index] = coord_2mass[0].ra.to_string(unit=u.hour, sep=' ')
                elif col.startswith('DEC'):
                    search[col][index] = coord_2mass[0].dec.to_string(unit=u.deg, sep=' ')

  
    search['IDS'][index] = saved_names
    return search

def query_simbad(date, coords, name=None, limit_G_mag=15, metadata=None, force_cm=False, verbose=False, pref_order=['HIP','HD','HR'], select='closest', is_moving=None):
    """
    Function that tries to query Simbad to find the object. 
    It first tries to see if the star name (optional argument) is resolved 
    by Simbad. If not it searches for the pointed position (ra and
    dec) in a cone of radius 10 arcsec. If more than a star is detected, it 
    takes the closest from the (ra,dec).
    Input:
        - date: an astropy.time.Time object (e.g. date = Time(header['DATE-OBS']), required.
            If more than one star is queried, date must be initialized as a Time object containing a list of dates.
        - name: string, numpy array or NoneType, optional. Name(s) of the source(s). If an array, 
            its len() must equal the number of dates/coords provided as input for 'date' and coords'.
            The array can also contain a mixture of valid names and elements = None or ''.
            If name=None, only coordinates are used. The same happens for array elements = None or ''.
            Default: None.
        - coords: a SkyCoord object, required. For instance, if we extract the keywords 
            of the fits files, we should use coords = SkyCoord(header['RA']*u.degree,header['DEC']*u.degree)
            Example of a valid input for a single star: SkyCoord('03h32m55.84496s -09d27m2.7312s', ICRS)
            If more than one star is queried, coords must be initialized as a SkyCoord object
            containing a list of coordinates.
        - limit_G_mag: int or float, optional. Limiting G (or, if not available, in order: V, J, R, U, B, I, J, H]) 
            magnitude beyond which we consider the star too faint to be the correct target. Default: 15.
        - metadata: dict, optional. Contains any additional information that one wants to pass in the output dictionary 
        - force_cm: bool, optional. If True, limit_G_mag is neglected, and cross-matches with no available  
            photometry in Simbad are retained. Default: False.
        - pref_order: list, optional. Catalogues, in descending order, to be
            preferentially stored into the 'simbad_BEST_NAME' keyword. Examples of valid
            catalogues: 'HIP', 'Gaia DR3', 'Gaia DR2', '2MASS', 'TYC', 'HD', 'HR' etc.
            You can also use 'proper_name' to pick star names such as 'Sirius', or 'bayer_designation'
            to pick names such as 'alp CMa'. Default: ['HIP','HD','HR'].
        - select: string, optional. Criterion to select the best cross-match when more than one star complies with selection
            criteria. It can be set to either 'closest' to pick the star with smaller separation to input coordinates,
            or to 'brightest' to pick the brightest star found in the FOV. Default: 'closest'.
        - is_moving: numpy array of dtype bool, optional. A boolean array indicating whether each star is a moving object or not.
            Moving objects will not be queried.
    Output:
        - a dictionary with the most interesting simbad keywords and the original 
            RA, DEC coordinates from the pointing position. A new keyword 'simbad_BEST_NAME' indicates the
            most adequate identifier according to the rule set by 'pref_order'.
    """
    
    # We check that every input is of correct type,
    # ensure that the first three inputs are arrays
    # and compute the number of input objects
    None_type = type(None)
    if type(name) not in [None_type,str,np.ndarray]:
        raise TypeError("'name' must be of type str, None or numpy.ndarray.")
    if type(coords) != SkyCoord:
        raise TypeError("'coords' must be a SkyCoord instance.")
    if type(date) != Time:
        raise TypeError("'date' must be a Time instance.")
    if (select!='closest') & (select!='brightest'): 
        raise ValueError("The keyword 'select' must be set to either 'closest' or 'brightest'.")
        
    if is_name_null(name): name = None
    if coords.isscalar: coords = SkyCoord(ra=[coords.ra],dec=[coords.dec],frame=coords.frame.name)
    if date.isscalar: date = Time([date.value])
        
    if isinstance(name,str):
        n_obj=1
        name=np.array([name],dtype=object)
    elif name is None:
        n_obj=len(coords)
    else:
        n_obj=len(name)
        name=np.array(name).astype(object)
        
    # array of moving objects is inizialized to zeros if not provided
    if is_moving is None: is_moving = np.zeros(n_obj,dtype=bool)
    is_there_any_moving_object = np.sum(is_moving)>0

    # some useful optional keywords are grouped in a dictionary
    useful_kwargs = {'limit_G_mag':limit_G_mag, 'verbose':verbose, 'pref_order': pref_order, 'select': select}
        
    # The output of the function is simbad_dico.
    # We first populate it with the initial values to pass 
    simbad_dico = {}
    simbad_dico['RA'] = coords.ra.to_string(unit=u.hourangle,sep=' ')
    simbad_dico['DEC'] = coords.dec.to_string(unit=u.degree,sep=' ')
    simbad_dico['DATE'] = date.iso
    if type(metadata) is dict:
        for key,val in metadata.items():
            simbad_dico[key] = val           

    if verbose: 
        print('-----------------------------')
        print('\nProgram started. No. of objects: {0}.'.format(n_obj))
        print('Step 1: checking input names.')

    # if no name is provided, an array of names is built based on coordinates
    skip_coord_check = False
    if name is None:
        if verbose:
            print(' No object names provided, trying to resolve them from coordinates...')
        name = []
        for i in range(n_obj):
            if verbose:
                print('  Star {0}/{1}. Input coordinates: (ra, dec) = ({2},{3}) '.format(i+1,n_obj,coords[i].ra.deg,coords[i].dec.deg))
            name_i = _query_simbad_from_coords(date[i],coords[i],**useful_kwargs)
            name.append(name_i)
        name = np.array(name).astype(object)
        skip_coord_check = True
        if verbose:
            print(' Done. \n')
            
    if verbose: print(' Checking accuracy of object names...')
        
    # some spelling checks are performed upon names
    for i in range(n_obj):
        coord = coords[i]
        name_i = str(name[i])
        if verbose & (skip_coord_check==False):
            print('  Star {0}/{1}. Input name: {2}.'.format(i+1,n_obj,name_i))
        if np.logical_and('47' in name_i,'tuc' in name_i.lower()):  # 47 Tuc
            name_i = 'Gaia EDR3 4689637789368594944'
        elif np.logical_and('3603' in name_i,'ngc' in name_i.lower()):
            name_i ='HD 97950B' # to be checked ! I'm not sure this is the AO star...
            if verbose: print('NGC 3603 case not implemented yet')
        elif np.logical_and('6380' in name_i,'ngc' in name_i.lower()):
            name_i = 'Gaia EDR3 5961801153907816832'
        elif np.logical_and('theta' in name_i.lower(),'ori' in name_i.lower()):
            name_i = 'tet01 Ori B' # theta Ori B1, the other astrometric calibrator often used
        elif name_i.lower()=='barnard':
            name_i ='Barnard star'
        elif name_i.lower().strip()=='no name':
            name_i = ''
        elif name_i.lower().strip()=='test':
            name_i = ''
        if name_i.startswith('NAME '):
            name_i=name_i[5:]
        if name_i.startswith('Vstar '):
            name_i=name_i[6:]
        if name_i.endswith(' System'):
            name_i=name_i[:-7]
        if name_i.startswith('HD'):
            name_i = 'HD '+((name_i.replace('HD','')).replace('_',' ')).replace(' ','')
        if name_i[-2:] in PLANET_ENDS:
            wrong_name = ''+name_i
            name_i=name_i[:-1]
            if verbose:
                print('  Warning! Wrong identifier detected: a planet ({0}) was used instead of its star ({1}). Fixing...'.format(wrong_name,name_i))
        if '2MASS' in name_i:
            match1 = re.match('2MASS*\s',name_i)
            match2 = re.match('2MASS*\s+J',name_i)
            if (match1 is not None) & (match2 is None):
                name_i = " J".join(name_i.split())
        if '=' in name_i:
            name_i = name_i.split('=')[0][:-1].strip()
        match1 = re.match("([a-zA-z][^_]{1,3}[0-9]*)\s*([A-Z]{1}[a-zA-z]{2}\s*)\s*\Z",name_i)
        match2 = re.match("([a-zA-z][^_]{1,3}[0-9]*)\s+([A-Z]{1}[a-zA-z]{2}\s*)\s*\Z",name_i)
        if (match1 is not None) & (match2 is None):
            name_i = (match1.groups()[0]+' '+match1.groups()[1]).strip()
        match1 = re.match("(^[a-zA-z][^_CD]{1,3})_*\s*([0-9]{1,20})\s+([0-9]{1,20})\s*\Z",name_i)
        match2 = re.match("(^[a-zA-z][^_CD]{1,3})_*\s*([0-9]{1,40})\s\Z",name_i)
        if (match1 is not None) & (match2 is None):
            name_i = ((match1.groups()[0]).strip()+' '+match1.groups()[1]+match1.groups()[2]).strip()
        match1 = re.match("([a-zA-z]{1}[0-9]{2})\s+([a-zA-z]{3})\s*\Z",name_i)
        if match1 is not None:
            name_i = '* '+(match1.groups()[0]+' '+match1.groups()[1]).strip()
        match1 = re.match("([V]{1})\s+([0-9]{2,10})\s+([a-zA-z]{3,20})",name_i)
        if match1 is not None:
            name_i = (match1.groups()[0]+match1.groups()[1]+' '+match1.groups()[2]).strip()
        match1 = re.match("([V]{1})\s+([a-zA-z]{1,10}\s*[0-9]*\s*[a-zA-z]{3,20})",name_i)
        if match1 is not None:
            name_i = (match1.groups()[0]+'* '+match1.groups()[1]).strip()

        if skip_coord_check: pass # if names were derived from coordinates, this step is skipped
        elif is_moving[i]:
            if verbose:
                print('   Object was indicated as a moving object, skipping...')
        elif is_name_null(name_i): # element with no name: we try to find it from coordinates
            if verbose:
                print('   No object name provided, trying to resolve it from coordinates...')
                print('   Input coordinates: (ra, dec) = ({0},{1}) '.format(coords[i].ra.deg,coords[i].dec.deg))
            name_i = _query_simbad_from_coords(date[i],coord,**useful_kwargs)
        else: # if an input name is present, we try to resolve it directly and see if input and object coordinates are consistent
            try: # if the input name is correctly resolved, the commands below are executed
                object_coordinate = SkyCoord.from_name(name_i.strip())
                separation = object_coordinate.separation(coord)
                if verbose:
                    print('   Object correctly resolved by Simbad.')
                    if separation < 0.1*u.arcsec:
                        print('   Separation between Simbad coordinates and input coordinates is {0:.2f} mas'.format(separation.mas))
                    elif separation < 1*u.arcmin:
                        print('   Separation between Simbad coordinates and input coordinates is {0:.2f} arcsec'.format(separation.arcsec))
                    elif separation < 1*u.deg:
                        print('   Separation between Simbad coordinates and input coordinates is {0:.2f} arcmin'.format(separation.arcmin))
                    else:
                        print('   Separation between Simbad coordinates and input coordinates is {0:.2f} deg'.format(separation.deg))
                if separation < SEARCH_RADIUS_ALT: # test passed: input object is the true object
                    if verbose:
                        print('   The object found is likely the target')
                    name_i = name_i.strip()
                else: # test failed: we try to resolve the name from coordinates
                    name_i = _query_simbad_from_coords(date[i],coord,force_verbose=False,**useful_kwargs)
                    if verbose:
                        print('   The object found is likely not the target.')
                        if name_i is None: # nothing found within SEARCH_RADIUS, returning None
                            print('   An attempt was made to only employ coordinates, but no object could be resolved. Input coordinates are likely wrong.')
                        else: # we found (at least) one source within SEARCH_RADIUS, we pick the most likely one
                            print('   Input object is likely wrong. An attempt was made to only employ coordinates, resolving the following object: '+name_i)
            except name_resolve.NameResolveError as e: # if the input name is not resolved, we try something else
                if verbose:
                    print('   '+str(e))
                if name_i[-2:] in BINARY_ENDS: 
                    name_i = name_i[:-1]
                    try: # the object had an unnecessary binary flag which was removed. If this solves the problem, the commands below are executed
                        object_coordinate = SkyCoord.from_name(name_i.strip())
                        separation = object_coordinate.separation(coord)
                        if verbose:
                            print('   The star appears to have an unnecessary binary flag in its name. Trying again without it:')
                            print('   New input object: ',name_i)
                            print('   Input object - Pointing separation is {0:.2f}'.format(separation))
                        if separation < SEARCH_RADIUS_ALT:
                            if verbose:
                                print('   The object found is likely the target')
                            name_i = name_i.strip()
                        else:
                            name_i = _query_simbad_from_coords(date[i],coord,force_verbose=False,**useful_kwargs)
                            if verbose:
                                print('   The object found is likely not the target.')
                                if name_i is None:
                                    print('   An attempt was made to only employ coordinates, but no object could be resolved. Input coordinates are likely wrong.')
                                else:
                                    print('   Input object is likely wrong. An attempt was made to only employ coordinates, resolving the following object: '+name_i)
                    except name_resolve.NameResolveError as e: # even removing the binary flag, the object is still not resolved. Returning None
                        if verbose:
                            print('   Again, input object {} was not recognized'.format(name_i.strip()))
                            print('   '+str(e))
                        name_i = None
                else: name_i = None
        
        # if the input name was changed during this step, we write it down explicitly
        if verbose & (str(name[i])!=name_i) & (is_moving[i]==False):
            print('   Note: Input name changed from {0} to {1}'.format(str(name[i]),name_i))

        name[i]=name_i
        
    if verbose:
        print('Step 1: done. \n \n')
    

    # at this point we have done our best to have a valid name recognized by Simbad
    # we are ready to retrieve all relevant information through a vectorized query

    customSimbad = Simbad()
    customSimbad.TIMEOUT = 600    

    # identifies components of binary systems, which require special care
    bin_flag = np.zeros(n_obj,dtype=str)
    for j in range(n_obj):
        if name[j] is not None:
            if name[j][-2:] in BINARY_ENDS:
                bin_flag[j]=name[j][-1]

    # name is changed from type object to type string to always enable string operations
    name = name.astype(str)

    if verbose:
        print('Step 2: querying object names on Simbad...')
        
    if n_obj==1:

        customSimbad.add_votable_fields('typed_id','ids','flux(U)','flux(B)','flux(V)','flux(R)',\
                                        'flux(I)','flux(G)','flux(J)','flux(H)',\
                                        'flux(K)','id(HD)','sp','otype','otype(V)','otype(3)',\
                                       'parallax','propermotions','ra(2;A;ICRS;J2000;2000)',\
                                     'dec(2;D;ICRS;J2000;2000)',\
                                     'ra(2;A;FK5;J{0:.3f};2000)'.format(date[0].jyear),\
                                     'dec(2;D;FK5;J{0:.3f};2000)'.format(date[0].jyear))
        search = customSimbad.query_objects(name)
    else:
        customSimbad.add_votable_fields('typed_id','ids','flux(U)','flux(B)','flux(V)','flux(R)',\
                                        'flux(I)','flux(G)','flux(J)','flux(H)',\
                                        'flux(K)','id(HD)','sp','otype','otype(V)','otype(3)',\
                                       'parallax','propermotions','ra(2;A;ICRS;J2000;2000)',\
                                     'dec(2;D;ICRS;J2000;2000)')           
        search = customSimbad.query_objects(name) # vectorized Simbad query
        search = _fix_binaries(search,name,bin_flag,customSimbad) # removes duplicate entries related to binaries

        if verbose: print(' Search ended.')
            
        # this array contains detailed info on how data were retrieved for each star
        program_comments = np.zeros(n_obj,dtype=object)

        # moving objects are now explicitly pointed out (and masked, if any previous info was present)
        if is_there_any_moving_object:
            search['MAIN_ID'][is_moving] = name[is_moving]
            search = Table(search, masked=True)
            for col in search.columns:
                if col in ['RA_2_A_ICRS_J2000_2000','DEC_2_D_ICRS_J2000_2000','OTYPE_3']:
                    search[col][is_moving] = ''
                elif col in ['OTYPE_V','OTYPE']:
                    search[col][is_moving] = 'Moving object'
                elif col in ['TYPED_ID','MAIN_ID','PMRA','PMDEC']: continue
                else: search[col].mask=is_moving
        
        
        # not every star was correctly resolved. Some additional steps are necessary
        
        # problem: not every star resolved by SkyCoord.from_name() is resolved by Simbad()
        # solution: we try again to solve stars starting from their coordinates
        search['index'] = np.arange(0,n_obj)
        mask1 = (search['OTYPE_V'] == 'Object of Unknown Nature')
        name_1 = []
        if np.sum(mask1)>0:
            if verbose:
                print(' Simbad was not able to resolve {0}/{1} input object names which were previously resolved by Sesame.'.format(np.sum(mask1),n_obj))
                print(' Trying again, only using coordinates and a search radius = {0}'.format(SEARCH_RADIUS))
            w,=np.where(mask1)
            for i in w:
                if verbose:
                    print('  Star {0}/{1}. Input coordinates: (ra, dec) = ({2},{3}) '.format(i+1,n_obj,coords[i].ra.deg,coords[i].dec.deg))
                name_1.append(_query_simbad_from_coords(date[i],coords[i],**useful_kwargs,enlarge_query=False))
            name_1 = np.array(name_1).astype(str)
            search2 = customSimbad.query_objects(name_1)
            search2 = _remove_duplicate_entries(search2) # removes duplicate entries related to binaries
            search2['index'] = w
            search = vstack((search[~mask1],search2))
            search.sort('index')
            del search['index']
            mask2 = (search['OTYPE_V'] == 'Object of Unknown Nature')
            if verbose:
                if np.sum(mask2)<np.sum(mask1):
                    print(' We were able to recover additional {0} targets.'.format(np.sum(mask1)-np.sum(mask2)))
                else:
                    print(' No additional target was recovered. Please carefully inspect input coordinates and names for these stars.')
        else: mask2 = mask1
        # a few comments on the results for these stars are saved
        program_comments[mask2] = 'Object not found. Either 1) wrong coordinates and, if provided, object name; 2) not a star'
        program_comments[~mask1] = 'Object properly resolved using coordinates and/or object name'
        program_comments[mask1 & ~mask2] = 'Object not resolved by Simbad, but correctly recovered through its coordinates'
        if is_there_any_moving_object: program_comments[is_moving] = 'Moving object'

        # problem: some stars are resolved but do not have associated photometry
        # solution: they are likely resolved binaries. Simbad lists entries for whole systems
        # and entries for individual components. If the components are far enough to be resolved by UBV-based surveys and 2MASS,
        # no photometry is present. We pick therefore the entries corresponing to 'A' components
        search['index'] = np.arange(0,n_obj)
        mask3 = np.ones(n_obj,dtype=bool)        
        for j in range(n_obj):
            break_photom = False
            i=0
            while (break_photom==False) & (i<len(FILTER_COLUMNS)):
                if np.isnan(search[FILTER_COLUMNS[i]][j])==False:
                    break_photom = True
                    mask3[j] = False
                i+=1
        mask3[mask2] = False # We are only interested in resolved stars
        if is_there_any_moving_object:
            mask3[is_moving] = False # We also exclude moving objects
        
        name_3 = []
        if np.sum(mask3)>0:
            if verbose:
                print(' No photometry found for {0}/{1} input objects which were correctly resolved.'.format(np.sum(mask3),n_obj))
                print(' Likely they are resolved binaries. Trying again, only using coordinates and a search radius = {0}'.format(SEARCH_RADIUS))
            w,=np.where(mask3)
            for i in w:
                if verbose:
                    print('  Star {0}/{1}. Input coordinates: (ra, dec) = ({2},{3}) '.format(i+1,n_obj,coords[i].ra.deg,coords[i].dec.deg))
                name_3.append(_query_simbad_from_coords(date[i],coords[i],**useful_kwargs,enlarge_query=False))
            name_3 = np.array(name_3).astype(str)
            name_3 = np.concatenate((name_3,['Vega']))            
            search4 = customSimbad.query_objects(name_3)[:-1]
            search4['index'] = w
            search = vstack((search[~mask3],search4))
            search.sort('index')
            del search['index']
            
            mask4 = np.ones(n_obj,dtype=bool)
            for j in range(n_obj):
                break_photom = False
                i=0
                while (break_photom==False) & (i<len(FILTER_COLUMNS)):
                    if np.isnan(search[FILTER_COLUMNS[i]][j])==False:
                        break_photom = True
                        mask4[j] = False
                    i+=1        
            mask4[mask2] = False # We are only interested in resolved stars
            if is_there_any_moving_object:
                mask4[is_moving] = False # We also exclude moving objects
            
            if verbose:
                if np.sum(mask4)<np.sum(mask3):
                    print(' We were able to recover additional {0} targets.'.format(np.sum(mask3)-np.sum(mask4)))
                else:
                    print(' No additional target was recovered. No photometry exists for these stars: be careful.')
        else: mask4 = mask3
        program_comments[mask4] = 'Object resolved in Simbad, but no photometry could be found'
        program_comments[mask3 & ~mask4] = 'Object represented a resolved binary system with no associated photometry. Replaced with its A component'
        
        # problem: some stars are simply missing in Simbad because they are too faint
        # solution: we try to recover them directly from Gaia DR3 and 2MASS using the VizieR resolver
        mask5 = ((search['OTYPE_V'] == 'Object of Unknown Nature') & (name!='None'))
        if np.sum(mask5)>0:
            if verbose:
                print(' Trying to solve {0}/{1} missing input objects on VizieR (Gaia DR3 + 2MASS).'.format(np.sum(mask3),n_obj))
            w,=np.where(mask5)
            for i in w:
                if verbose:
                    print('  Star {0}/{1}. Input name = {2} '.format(i+1,n_obj,name[i]))
                search = _vizier_resolver(name[i],coords[i],search,i)
  
            mask6 = ((search['OTYPE_V'] == 'Object of Unknown Nature') & (name!='None'))
            if verbose:
                if np.sum(mask6)<np.sum(mask5):
                    print(' We were able to recover additional {0} targets.'.format(np.sum(mask5)-np.sum(mask6)))
                else:
                    print(' No additional target was recovered using VizieR.')
        else: mask6 = mask5
        program_comments[mask6] = 'Object not resolved by either Simbad or VizieR'
        program_comments[mask5 & ~mask6] = 'Object not resolved by Simbad but recovered in VizieR'
        

    # At this point, data for every star should have been collected in one way or another
    
    if verbose:
        print('Step 2: done. \n \n')
        print('Step 3: notes on problematic objects')
        
    # the few missing objects will be printed at this point
    # here we are computing object coordinates at the observing epoch starting from J2000 coordinates and proper motions
    # a systematic error < 2*parallax is present due to neglect of parallactic effects
    RA1, DEC1 = [], []
    for i in range(n_obj):
        RA0 = search['RA_2_A_ICRS_J2000_2000'][i]
        DEC0, PMRA0, PMDEC0 = search['DEC_2_D_ICRS_J2000_2000'][i],\
        search['PMRA'][i], search['PMDEC'][i]
        if ((type(name[i])==np.ma.core.MaskedConstant) | (type(RA0)==np.ma.core.MaskedConstant) | (type(PMRA0)==np.ma.core.MaskedConstant) | (type(PMDEC0)==np.ma.core.MaskedConstant)):
            RA1.append('')
            DEC1.append('')
        elif ((name[i]!='None') & (RA0!='') & (np.isnan(PMRA0)==False) & (np.isnan(PMDEC0)==False)):
            dt=(Time(date[i])-Time('J2000.0')).jd*u.day.to(u.yr)
            coo=SkyCoord(ra=RA0,dec=DEC0,unit=(u.hourangle,u.deg),frame='icrs')
            coo1=SkyCoord(ra=coo.ra+PMRA0*u.mas/u.yr*dt*u.yr,dec=coo.dec+PMDEC0*u.mas/u.yr*dt*u.yr,frame='icrs')
            coo1_string = (coo1.transform_to(FK5)).to_string('hmsdms').split(' ')
            ra1_string = coo1_string[0].replace('h',' ').replace('m',' ').replace('s','').split('.')
            dec1_string = coo1_string[1].replace('d',' ').replace('m',' ').replace('s','').split('.')
            RA1.append(ra1_string[0]+str(round(float('0.'+ra1_string[1]), 4))[1:])
            DEC1.append(dec1_string[0]+str(round(float('0.'+dec1_string[1]), 4))[1:])
        elif (name[i]!='None') & (RA0==''):
            RA1.append('')
            DEC1.append('')
            if verbose & is_moving[i]==False:
                print('  Object {} was resolved by the CDS, but was not found on Simbad'.format(name[i]))
        else:
            RA1.append('')
            DEC1.append('')

    search['RA_2_A_FK5_obstime']=RA1
    search['DEC_2_D_FK5_obstime']=DEC1
    
    if is_there_any_moving_object:
        for col in search.columns:
            if col in ['PMRA','PMDEC','MAIN_ID']: search[col].mask[is_moving] = True

    if verbose:
        print('Step 3: done. \n \n')
        print('Step 4: photometric vetting')
    
    
    # Here we find the most adequate identifiers according to user-defined rules
    best_names=np.array(_get_best_id(search,pref_order))
    if is_there_any_moving_object: best_names[is_moving] = ''
    search['BEST_NAME'] = best_names
    search['BIN_FLAG'] = bin_flag
    search['program_comments'] = program_comments
    
    
    
    
    del search['IDS']

    
    # photometric vetting and creation of simbad_dico
    mask7 = (search['OTYPE_V'] == 'Object of Unknown Nature')
    nb_stars, i = 0, 0
    if n_obj==1:
        if search is not None:
            while (nb_stars==0) & (i<len(FILTER_COLUMNS)): # if the star is fainter than that, it's likely not the one we are looking for
                validSearch = search[search[FILTER_COLUMNS[i]]<limit_G_mag]
                nb_stars = len(validSearch)
                i+=1
            # sometimes no photometry is available on Simbad(). Depending on 
            # the truth value of 'force_cm', one can accept the cross-matched object or reject it 
            if (nb_stars==0) & (force_cm):
                if verbose:
                    print(' No photometry available for this star, check it carefully.')
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
                print(' Something went wrong, there are {0:d} valid stars '.format(nb_stars))
            return None

    else:
        if search is not None:
            mask = (search[FILTER_COLUMNS[i]]<-30).filled(False)
            while (nb_stars<len(search)) & (i<len(FILTER_COLUMNS)): # if the star is fainter than that, it's likely not the one we are looking for
                mask += ((search[FILTER_COLUMNS[i]]<limit_G_mag)).filled(False)
                nb_stars = np.sum(mask)
                i+=1
            if is_there_any_moving_object: mask[is_moving] = True
            if (nb_stars<len(search)) & (force_cm):
                n_stars_wo_cm = n_obj-nb_stars-np.sum(mask7)-np.sum(is_moving)
                if verbose:
                    if n_stars_wo_cm>0:
                        print(' force_cm is set to True, so the cross-match was considered valid for {0} stars even though they had no photometry or photometry below threshold. Be careful.'.format(n_stars_wo_cm))
                    else:
                        print(' No resolved star had missing photometry, so this step was skipped.')
                        
                nb_stars=len(search)
        else: nb_stars=0

        if nb_stars < len(search):
            n_stars_wo_cm = n_obj-nb_stars-np.sum(mask7)-np.sum(is_moving)
            if n_stars_wo_cm>0:
                if verbose:
                    print(' No photometry was found for {0} stars, so they were rejected.'.format(n_stars_wo_cm))
                search = Table(search, masked=True)
                for col in search.columns:
#                    if col!='TYPED_ID': search[col].mask=~mask
                    if col!='TYPED_ID': search[col].mask[~mask] = True
                program_comments[(~mask7) & (~mask)] = 'Object info deleted because either 1) no photometry on Simbad; 2) dimmer than limit_G_mag'
                search['program_comments']= program_comments
            else:
                if verbose:
                    print(' No resolved star had missing photometry, so this step was skipped.')
       
        simbad_dico = populate_simbad_dico(search,None,simbad_dico)
        simbad_dico  = add_separation_between_pointing_current_position(coords,simbad_dico)
        
        if verbose:
            print('Step 4: done. Program ended.\n\n')
            
        return simbad_dico
        

def populate_simbad_dico(simbad_search_list,i,simbad_dico):
    """
    Method not supposed to be used outside the query_simbad method
    Given the result of a simbad query (list of simbad objects), and the index of 
    the object to pick, creates a dictionary with the entries needed.
    """
    
    for key in simbad_search_list.keys():
        value = simbad_search_list[key] if i==None else simbad_search_list[key][i]
        if key in ['MAIN_ID','BEST_NAME','SP_TYPE','ID_HD']:
            try:
                simbad_dico['simbad_'+key] = np.array(value.filled(''))
            except AttributeError:
                simbad_dico['simbad_'+key] = np.array(value)                    
        elif key=='OTYPE':
            try:
                simbad_dico['simbad_'+key] = np.array(value.filled('Unknown'))
            except AttributeError:
                simbad_dico['simbad_'+key] = np.array(value)
        elif key=='OTYPE_V':
            try:
                simbad_dico['simbad_'+key] = np.array(value.filled('Object of Unknown Nature'))
            except AttributeError:
                simbad_dico['simbad_'+key] = np.array(value)
        elif key=='OTYPE_3':
            try:
                simbad_dico['simbad_'+key] = np.array(value.filled('?'))
            except AttributeError:
                simbad_dico['simbad_'+key] = np.array(value)
        elif key.startswith('FLUX_'):
            try:
                simbad_dico['simbad_'+key] = np.array(value.filled(np.nan),dtype=float)
            except AttributeError:
                simbad_dico['simbad_'+key] = np.array(value,dtype=float)
        elif key in ['PMDEC','PMRA']:
            try:
                simbad_dico['simbad_'+key] = np.array(value.filled(np.nan),dtype=float)
            except AttributeError:
                simbad_dico['simbad_'+key] = np.array(value,dtype=float)
        elif key.startswith('RA_2_A_FK5_'): 
            simbad_dico['simbad_RA_current'] = value
        elif key.startswith('DEC_2_D_FK5_'): 
            simbad_dico['simbad_DEC_current'] = value
        elif key in ['RA','DEC']:
            simbad_dico['simbad_'+key+'_ICRS'] = value
        elif key=='program_comments':
            simbad_dico[key] = value
                
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
    testCoord = SkyCoord(ra,dec)
    date = Time('2017-01-01T02:00:00.0')
    print("Let's query a random coordinates ra={0:s} dec={1:s} with the name {2:s} and see what's happening\n".format(testCoord.ra,testCoord.dec,name))
    test=query_simbad(date,testCoord,name='eps Eri',limit_G_mag=15)
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
    ra = 6.01*u.degree
    dec = -72.09*u.degree
    name='47 Tuc'
    testCoord = SkyCoord(ra,dec)
    date = Time('2017-01-01T02:00:00.0')
    print("Let's query 47 Tuc at ra={0:s} dec={1:s} with the name {2:s} and see what's happening\n".format(testCoord.ra,testCoord.dec,name))
    test=query_simbad(date,testCoord,name=name,limit_G_mag=15,verbose=True)
    
    print('\n\n','-'*20)
    h = fits.getheader(os.path.join(path_data,'SPHER.2019-04-01T03-39-17.958IRD_SCIENCE_DBI_RAW.fits'))
    print("Let's query a target from a real SPHERE header\n")
    test = query_simbad_from_header(h)
    
    print('\n\n','-'*20)
    h = fits.getheader(os.path.join(path_data,'SPHER.2019-02-25T03-55-45.738ZPL_SCIENCE_IMAGING_RAW.fits'))
    print("Let's query a target from a real SPHERE header (a moving target in this case) \n")
    test = query_simbad_from_header(h)
