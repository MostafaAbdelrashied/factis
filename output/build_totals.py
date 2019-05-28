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
cmd_parentfolder = '/'.join((os.path.realpath(os.path.dirname(inspect.getfile( inspect.currentframe() )))).split('/')[:-1])
if cmd_parentfolder not in sys.path:
    sys.path.insert(0, cmd_parentfolder)
from genpy.datacrunching import UNIXDataFrameColumn2DataIndex, DATAinput2TimeSeries, csv_to_dict
from paths import input_folder, output_folder, parsed_files_dir, generic_folder, bat_spec_dir

####-
myFolder =  (os.path.realpath(os.path.dirname(inspect.getfile( inspect.currentframe() ))))

### Functions

def calculate_energy(Data_Frame, list_of_column_names, period):
    """Creates a dataframe with the accumulated energy from the columns defined in the list"""
    # period = period_between_steps(Data_Frame)
    # period
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

#def period_between_steps(input_df):
#    """Calculates the step value"""
#    numberofvalues = float(len(input_df.index))
#    last_timestamp = input_df.hoy[numberofvalues-1]
#    first_timestamp = input_df.hoy[0]
#    period = (last_timestamp-first_timestamp)/(numberofvalues*3600)  # period in hours
#    return period

def selfconsumption(df):
    """Calculates the self-consumption and the self-sufficiency (autarky)"""
    #print df.PV_local_kWh, df.PV1_Pel_kWh
    self_consumption_ratio = float(df.PV_local_kWh) / float(df.PV1_Pel_kWh)
    autarky_ratio = 1 - (df.grid_load_supply_kWh / df.Load_Pel_kWh)
    df['Self_Consumption'] = self_consumption_ratio
    df['Self_Sufficiency'] = autarky_ratio
    return df

def limitedfeedinpower(df, maxpercent, nominalpowerofPV, period):
    """Creates a dictionary with Eloss and Efeed based on max percentual feed in from the nominal Power."""
    #period = period_between_steps(Data_Frame)

    maxpercent = float(maxpercent)
    nominalpowerofPV = float(nominalpowerofPV)
    exceedingPV_grid_feed = df.PV_grid_feed[df.PV_grid_feed > (maxpercent*nominalpowerofPV)]
    Eloss = (exceedingPV_grid_feed.sum() - len(exceedingPV_grid_feed)*nominalpowerofPV*maxpercent)*period
    Efeed = (df.PV_grid_feed.sum()*period) - Eloss
    return {"Eloss" : Eloss, "Efeed" : Efeed}

# CORE

if __name__ == "__main__":
    from sys import argv
    specDict = csv_to_dict("{}/spec.txt".format(myFolder))
    name_short = specDict['name'][:-4]
    if specDict['fileorfolder'] == 'folder':
        spec_filename = 'res_{szenname}_{inputfile1}{inputfile2}.dat'.format(**specDict)        
    else:
        spec_filename = 'res_{szenname}_{name}'.format(**specDict)
    print ('\nCalculating totals from: {}'.format(spec_filename))
    Data_Frame = pd.read_csv('{}/{}'.format(myFolder, spec_filename),  delim_whitespace = 1)
    
    list_of_column_names_power = [
                                  'Load_Pel', 'PV1_Pel', 
                                  'grid_load_supply', 'PV_grid_feed'
                                 ]
    list_of_max = calculate_max_power(Data_Frame, list_of_column_names_power)

    #list_of_limit =  limitedfeedinpower(Data_Frame, 0.6, '{scale_pv}'.format(**specDict)) delte me please

    list_of_column_names_energy = [
                                   'Load_Pel', 'PV1_Pel', 'grid_load_supply', 
                                   'PV_local', 'PV_grid_feed', 'PV_self', 
                                   'PV_direct', 'sto_in', 'sto_out', 
                                   'bat_in', 'bat_out'
                                  ]

    list_of_energy = calculate_energy(Data_Frame, list_of_column_names_energy,  float(specDict['step']))
    
    list_of_totals = list_of_max.join(list_of_energy)
    output_file = str('totals_{0}'.format(spec_filename))
    list_of_totals = selfconsumption(list_of_totals)

    dictoflimit =  limitedfeedinpower(Data_Frame, 0.6, '{scale_pv}'.format(**specDict),  float(specDict['step']))
    
    specDict_ds = pd.Series(specDict)
    specDict_df = pd.DataFrame(specDict_ds)
    specDict_df = specDict_df.transpose()
    #specDict_df = specDict_df.set_index("name")
     

    list_of_totals['Eloss'] = dictoflimit["Eloss"]
    list_of_totals['Efeed'] = dictoflimit["Efeed"]
    output_df = pd.concat([specDict_df, list_of_totals], axis=1)
    output_df =  output_df.set_index("szenname")
    output_df = output_df.reindex_axis(sorted(output_df.columns), axis=1)

    output_df.to_csv('{}/{}'.format(myFolder, output_file), sep = '\t', float_format='%.3f')
    print ('File with totals was written to: {}/{}\n'.format(output_folder, output_file))
