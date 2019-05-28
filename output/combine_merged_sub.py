#!/usr/bin/python
### combine parsed subfiles output to new File
import csv
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from datetime import datetime, timedelta

#### local/relativ modul import

import sys, inspect, os
cmd_parentfolder = '/'.join((os.path.realpath(os.path.dirname(inspect.getfile( inspect.currentframe() )))).split('/')[:-1])
if cmd_parentfolder not in sys.path:
    sys.path.insert(0, cmd_parentfolder)
from ampl import  UNIXDataFrameColumn2DataIndex, subPartinDF, csv_to_ts interpret_spec
from combine_parsed_sub import combineResults

from genpy.datacrunching import UNIXDataFrameColumn2DataIndex, DATAinput2TimeSeries, csv_to_dict, n_comments
from genpy.systools import sorted_list_of_files, delete_files_in_folder, build_folder_structure, folder_or_file 

####-
myFolder =  (os.path.realpath(os.path.dirname(inspect.getfile( inspect.currentframe() ))))


# CORE
"""
def combineResults(folder, period = 'D', length = 1, start= 0, end = 0):
    '''combine all the files located in the folder into one single dataframe.
    '''
    sorted_file_list = getSortedListofFilesinFolder(folder)
    if end != 0: 
        sorted_file_list = sorted_file_list[:end]
    if start != 0: 
        sorted_file_list = sorted_file_list[start:] 
    number_of_files = len(sorted_file_list)
    firstFile = sorted_file_list.pop(0)
    print '\nCombining files\nThe first file in the folder is called: {0}\nThere are {1} files to be combined.'.format(firstFile, number_of_files)
    path = '{0}/{1}'.format(folder, firstFile)
    
    global_df = subPartinDF(path, period, length)
    for fileName in sorted_file_list:
        path = '{0}/{1}'.format(folder, fileName)
        next_files_sub_part =  subPartinDF(path, period, length)
        try: 
            global_df = global_df.append(next_files_sub_part, ignore_index = False)
        except AttributeError:
            print "The KeyError caused an AttributeError!"
    return global_df



def subPartinDF(path, period='D', period_numbers=1):
    '''Get first period of file by converting it to pandas time series. 
    Knows days hours minutes and seconds. Be careful with the timezone, since duplicated timeseries (e.g. summertime change) can rise an error.
    '''
    period_numbers = float(period_numbers)
    df_ts = pd.read_csv(path, delim_whitespace=True, parse_dates=0, index_col="time")
       
    first_time_step = df_ts.index[0]
    periodDict = {'D': 'days', 'H': 'hours', 'M': 'minutes', 'S': 'seconds'}
    last_time_step = first_time_step + timedelta(**{periodDict[period]: period_numbers}) - timedelta(seconds=1)
    try:
        subPart = df_ts.ix[str(first_time_step): str(last_time_step)]
    except KeyError:
        ErrorM = 'There was a KeyError in file {0} in the timeperiod between {1} and {2}! Maybe it is caused by day time shift e.g. summer time to winter time and therefore duplicate indexes.'.format(path, first_time_step, last_time_step)
        print ErrorM
        subPart = ErrorM
    return subPart




"""


if __name__ == "__main__":
    from sys import argv
    specDict = interpret_spec("{}/spec.txt".format(myFolder))
    global_df = combineResults('{}/merged'.format(myFolder),  specDict['period'],  specDict['length'], 0, 0, 'hoy_x')
    name_short = specDict['name']
    final_filename = 'res_{szenname}_{0}_{length}_{overlap}_{period}_merged.dat'.format(name_short, **specDict)
    global_df.to_csv('{}/{}'.format(myFolder, final_filename), sep = ' ')
    print 'The combined file is in: {0}/{1}\n'.format("output", final_filename)
