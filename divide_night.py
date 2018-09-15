#!/Users/jmilli/anaconda/envs/python36/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 15:27:18 2018

This script asks you to enter the date and then gives in output the time
of sunset, sunrise, twilights, middle of the night as well as any fractions
of the nights (multiple of 0.1night). 

The goal is to ease the task of the night astronomer when a visitor has
only a fraction of the night by avoiding to make mistakes. 

It makes use of numpy and astral package, and is set up for Paranal only.

@author: jmilli
"""

import numpy as np
import astral, datetime


paranal_astral = astral.Location(info=("Paranal", "Chile", -24.6268,-70.4045, "Etc/UTC",2648.0))
paranal_astral.solar_depression = "astronomical"
# "civil" means 6 degrees below the horizon, is the default
# value for computing dawn and dusk. "nautical" means 12 degrees
# and "astronomical" means 18 degrees


date_str = input('Enter the date (use the iso format 2018-10-13)or just press enter for today:\n')
if date_str == '':
    day = datetime.date.today()
else:
    try:
        day_array = date_str.split('-')
        day = datetime.date(int(day_array[0]),int(day_array[1]),int(day_array[2]))
    except:
        print("The date was not understood. We assume you mean today's date")
        day = datetime.date.today()

print('The date of observation is set to {0:s}\n'.format(day.isoformat()))

result_today    = paranal_astral.sun(date=day)
result_tomorrow = paranal_astral.sun(date=day + datetime.timedelta(1))
midnight        = result_today['dusk']+(result_tomorrow['dawn']-result_today['dusk'])/2.

print('Sunset                   : {0:s}'.format(result_today['sunset'].isoformat()))
print('End of evening twilight  : {0:s}'.format(result_today['dusk'].isoformat()))
print('Middle of the night      : {0:s}'.format(midnight.isoformat()))
print('Start of morning twilight: {0:s}'.format(result_tomorrow['dawn'].isoformat()))
print('Sun rise                 : {0:s}\n'.format(result_tomorrow['sunrise'].isoformat()))

fractions_night = np.linspace(0.1,0.9,9)
fractions_night = np.delete(fractions_night,4)
description_fractions_night = ['{0:.1f}n'.format(frac) for frac in fractions_night]
description_fractions_halfnight = ['{0:.1f}H{1:d}'.format(np.mod(2*frac,1),int(2*frac+1)) for frac in fractions_night]

for i,description in enumerate(description_fractions_halfnight):
    t = result_today['dusk']+(result_tomorrow['dawn']-result_today['dusk'])*fractions_night[i]
    print('{0:s} = {1:s} :  {2:s}'.format(description,description_fractions_night[i],t.isoformat()))
