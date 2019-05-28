#!/usr/bin/python
# rholling
# Creates sub files from AMPL input as AMPL input
# [1] The filename
# [2] The length of the result to be used.
# [3] The overlap defines the overlap after the length of the result to be used.
# [4]  The kind of Period: D for days, H for Hours, M for minute and S for seconds
# Example:
# build_sub.py AMPL_2012_Referenzdatensatz.dat 1 0.5 D

from __future__ import print_function

import pandas as pd
from datetime import timedelta
import sys
from sys import argv
# import inspect
import os

# cmd_parentfolder = "/".join((os.path.realpath(
#     os.path.dirname(inspect.getfile(inspect.currentframe())))).split('/')[:-1])
# if cmd_parentfolder not in sys.path:
#     sys.path.insert(0, cmd_parentfolder)
from genpy.datacrunching import DATAinput2TimeSeries
from genpy.systools import delete_files_in_folder
from genpy.systools import build_folder_structure


def count_periods(name, period, number=1):
    """
    Count number of (number times) periods (day='D', Hour = 'H') in pandas DataFrame.
    """

    startPeriod = pd.Period(getattr(name, 'index')[0], freq=period)
    endPeriod = pd.Period(getattr(name, 'index')[-1], freq=period)
    amount = int(endPeriod - startPeriod)
    return amount / number


def global2subfile(pd_ts, filename, length=1, overlap=0.5, period='D', startp=0, endp=None):
    """
    Build from global DataFrame subFiles defined by length of sub files, start and end.
    [pd_ts]     : The pd_ts of the pandas.DataFrame with the data, the time series index and the colum names.
    [filename] : The filename
    [length]   : The length of the result to be used.
    [overlap]  : The overlap defines the overlap after the length of the result to be used.
    [period]   : The kind of Period: D for days, H for Hours, M for minute and S for seconds
    [startp]   : The default value for start is the first available (value=0).
    [endp]     : If None is given the default value for end is the last available (value calculated).
    """

    if endp is None:
        endp = count_periods(pd_ts, period, length)
        print('end was calculated by number of {0} in File: {1}'.format(period, endp))

    sub_folder = '{0}_{1}_{2}_{3}_subFiles'.format(
        filename, length, overlap, period)
    _temp_folderpath, _sub_file = os.path.split(sub_folder)
    if build_folder_structure(['{sf}'.format(sf=sub_folder)]):
        print('The Folder', sub_folder, 'exists already. Will delete all files in Folder!')

        delete_files_in_folder(sub_folder)
    periodDict = {'D': 'days', 'H': 'hours', 'M': 'minutes', 'S': 'seconds'}
    lperiod = str(periodDict[period])
    for x in range(startp, int(endp + 1)):
        file_start = getattr(pd_ts, 'index')[0]
        file_start = file_start.replace(second=0)
        loc_start = file_start + timedelta(**{lperiod: x * length})
        loc_end = file_start + \
                  timedelta(**{lperiod: (x + 1) * length}) + \
                  timedelta(**{lperiod: overlap}) - timedelta(seconds=1)

        sub_file = '{0}_{1}'.format(x, _sub_file)
        subFile_in_subFolder = os.path.join('{0}', '{1}').format(sub_folder, sub_file)

        with open(subFile_in_subFolder, 'w') as SFinSF:
            if int(x) == endp:
                name_sub = pd_ts[loc_start:]
            else:
                sys.stdout.write(
                    '\rActual cut dates: {0} --> {1}'.format(loc_start, loc_end))
                sys.stdout.flush()
                name_sub = pd_ts[loc_start:loc_end]

            timestep_number = int(len(name_sub.index))
            name_sub.loc[:, "param:"] = range(1, timestep_number + 1)
            name_sub = name_sub.set_index("param:")
            name_sub.to_csv(SFinSF, sep=" ")
            endline = ';\nparam T := {0};'.format(timestep_number)
            print(endline, file=SFinSF)


if __name__ == "__main__":
    print("Read file in pandas DataFrame and convert it to pandas timeseries.")
    param_dict = {'file_path': 'input/energycharts_downloads_price_2018_en_30min.dat',  # .dat file path
                  'control_horizon': 24.,  # Control Horizon value (Float)
                  'overlap': 12.,  # Overlap value (Float) //// Predication horizon = Control Horizon + Overlap
                  'type': 'H'}
    print('\n')
    ampl_ts = DATAinput2TimeSeries(param_dict['file_path'], 'hoy', cutendlines=2)
    print("Cut in sub files.")
    global2subfile(ampl_ts, param_dict['file_path'], param_dict['control_horizon'], param_dict['overlap'],
                   param_dict['type'])
    exit()
    # print ('\n')
    # ampl_ts = DATAinput2TimeSeries(argv[1], 'hoy', cutendlines=2)
    # print ("Cut in sub files.")
    # global2subfile(ampl_ts, argv[1], float(argv[2]), float(argv[3]), argv[4])
