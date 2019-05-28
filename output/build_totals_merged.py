#!/usr/bin/python
### Calculate energy totals in kWh and maximum power ir kW from file defined in spec.txt
### 
from __future__ import print_function

from os import path, makedirs
import pandas as pd
import numpy as np
import os
####+ local/relativ modul import

import sys, inspect, os
# cmd_parentfolder = '/'.join((os.path.realpath(os.path.dirname(inspect.getfile( inspect.currentframe() )))).split('/')[:-1])
# if cmd_parentfolder not in sys.path:
#     sys.path.insert(0, cmd_parentfolder)
from genpy.datacrunching import csv_to_dict
from paths import input_folder, output_folder, parsed_files_dir, generic_folder, bat_spec_dir



####-
myFolder =  (os.path.realpath(os.path.dirname(inspect.getfile( inspect.currentframe() ))))
### Functions

def calculate_energy(Data_Frame, list_of_column_names, period):
    """Creates a dataframe with the accumulated energy from the columns defined in the list"""
    #period = period_between_steps(Data_Frame)
    period
    column_name = list_of_column_names.pop(0)
    a = getattr(Data_Frame, column_name).sum() * (period)
    column_name_kWh = '{}_kWh'.format(column_name)
    df_of_values = pd.DataFrame([a], columns=[column_name_kWh]) 

    for column_name in list_of_column_names:
        a = getattr(Data_Frame, column_name).sum() * (period)
        column_name_kWh = '{}_kWh'.format(column_name)
        df_of_values[column_name_kWh] = a
    return df_of_values


def calculate_max_power(Data_Frame, list_of_column_names):
    """Creates a dataframe with the maximum power from the columns defined in the list"""
    column_name = list_of_column_names.pop(0)
    a = getattr(Data_Frame, column_name).max() 
    column_name_max_kW = '{}_max_kW'.format(column_name)
    df_of_values = pd.DataFrame([a], columns=[column_name_max_kW])

    for column_name in list_of_column_names:
        a = getattr(Data_Frame, column_name).max()
        column_name_max_kW = '{}_max_kW'.format(column_name)
        df_of_values[column_name_max_kW] = a 

    return df_of_values 

def selfconsumption_merged(df):
    """Calculates the self-consumption and the self-sufficiency (autarky)"""
    self_consumption_ratio = df.PV_local_x_kWh / df.PV1_Pel_x_kWh
    autarky_ratio = 1 - (df.grid_load_supply_x_kWh / df.Load_Pel_x_kWh)
    df['Self_Consumption'] = self_consumption_ratio
    df['Self_Sufficiency'] = autarky_ratio
    return df

# CORE

if __name__ == "__main__":
    # from sys import argv
    specDict = csv_to_dict("{}/spec.txt".format(myFolder))
    name_short = specDict['name'][:-4]
    if specDict['fileorfolder'] == 'folder':
        spec_filename = 'res_{szenname}_{inputfile1}{inputfile2}{realdatafile}_merged.dat'.format(**specDict)        
    else:
        spec_filename = 'res_{szenname}_{name}'.format(**specDict)
    #if specDict['fileorfolder'] == 'folder':
    #    spec_filename = 'res_{szenname}_{0}_{length}_{overlap}_{period}_merged.dat'.format(name_short, **specDict)        
    #else:
    #    spec_filename = 'res_{szenname}_{0}_merged.dat'.format(name_short, **specDict)
    print ('\nCalculating totals from: {}'.format(spec_filename))
    Data_Frame = pd.read_csv('{}/{}'.format(myFolder, spec_filename),  delim_whitespace = True)
    
    list_of_column_names_power = [
                                  'Load_Pel_x', 'PV1_Pel_x', 
                                  'grid_load_supply_x', 'PV_grid_feed_x'
                                 ]
    list_of_max = calculate_max_power(Data_Frame, list_of_column_names_power)

    list_of_column_names_energy = [
                                   'Load_Pel_x', 'PV1_Pel_x', 'grid_load_supply_x', 
                                   'PV_local_x', 'PV_grid_feed_x', 'PV_self_x', 
                                   'PV_direct_x', 'sto_in_out_x'
                                  ]

    list_of_energy = calculate_energy(Data_Frame, list_of_column_names_energy, float(specDict['step']))
    
    list_of_totals = list_of_max.join(list_of_energy)
    output_file = str('totals_{0}'.format(spec_filename))
    list_of_totals = selfconsumption_merged(list_of_totals)
    list_of_totals.to_csv('{}/{}'.format(myFolder, output_file), sep = ' ')
    print ('File with totals was written to: {}/{}\n'.format(output_folder, output_file))


