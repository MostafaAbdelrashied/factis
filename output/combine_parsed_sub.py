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
from ampl import  subPartinDF, csv_to_ts, interpret_spec
from genpy.datacrunching import UNIXDataFrameColumn2DataIndex, DATAinput2TimeSeries, csv_to_dict, n_comments
from genpy.systools import sorted_list_of_files, delete_files_in_folder, build_folder_structure, folder_or_file 

####-
myFolder =  (os.path.realpath(os.path.dirname(inspect.getfile( inspect.currentframe() ))))

# FUNCTIONS 



def combineResults(folder, period = 'D', length = 1, start= 0, end = 0, unix_column = 'hoy'):
    """combine all the files located in the folder into one single dataframe.
    """
    sorted_file_list = sorted_list_of_files(folder)
    if end != 0: 
        sorted_file_list = sorted_file_list[:end]
    if start != 0: 
        sorted_file_list = sorted_file_list[start:] 
    number_of_files = len(sorted_file_list)
    firstFile = sorted_file_list.pop(0)
    print '\nCombining files\nThe first file in the folder is called: {0}\nThere are {1} files to be combined.'.format(firstFile, number_of_files)
    path = '{0}/{1}'.format(folder, firstFile)
    global_df = subPartinDF(path, period, length, unix_column)
    for fileName in sorted_file_list:
        path = '{0}/{1}'.format(folder, fileName)
        next_files_sub_part =  subPartinDF(path, period, length, unix_column)
        try: 
            global_df = global_df.append(next_files_sub_part, ignore_index = False)
        except AttributeError:
            print "The KeyError caused an AttributeError!"
    return global_df
    
# CORE

if __name__ == "__main__":
    from sys import argv
    specDict = interpret_spec("{}/spec.txt".format(myFolder))
    global_df = combineResults('{}/parsed'.format(myFolder),  specDict['period'],  specDict['length'])
    name_short = specDict['name']#[:-4]
    final_filename = 'res_{szenname}_{0}_{length}_{overlap}_{period}.dat'.format(name_short, **specDict)
    global_df.to_csv('{}/{}'.format(myFolder, final_filename), sep = ' ')
    print 'The combined file is in: {0}/{1}\n'.format("output", final_filename)
