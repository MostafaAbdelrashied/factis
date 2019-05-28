"""
!/usr/bin/python
 mabdelra 25.07.2018
 [1] Input data file or folder with subfiles. For example: AMPL_2012_Referenzdatensatz.dat
 [2] Loop start. For example: first day of the year --> 0 [Default: 0]
 [3] Loop end. For example: last day of the year --> 0. Day 15 of the year --> 15 [Default: 0]
 [4] Correction. Binary digit to define if the optimization result is merged with
    the input data and the real battery operation is calculated. 0 = No, 1 = Yes. [Default = 0]
 [5] Battery Capacity: Used to write the battery capacity in the results file name. [Default = ""]
 """

from __future__ import print_function

import csv
from subprocess import call
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from os import listdir, makedirs, path, unlink
from os.path import isfile, join, exists, isdir
import pytz

# + local/relativ modul import
import sys
import inspect
import os
import re

# cmd_folder = os.path.realpath(
#     os.path.dirname(inspect.getfile(inspect.currentframe())))
# if cmd_folder not in sys.path:
#     sys.path.insert(0, cmd_folder)
# os.getcwd(cmd_folder)
# import paths to different files from paths.py
from paths import input_folder, output_folder, parsed_files_dir, generic_folder, merged_files_dir, bat_spec_dir, bat_dat_filename, period_dat_filename

# genpy imports
from genpy.datacrunching import UNIXDataFrameColumn2DataIndex, DATAinput2TimeSeries, csv_to_dict, n_comments
from genpy.systools import sorted_list_of_files, delete_files_in_folder, build_folder_structure, folder_or_file

# -
myFolder = (
    os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe()))))

# Functions


def readFile2DFts(path, unix_column = 'hoy'):
    """
    Reads CSV with whitespace delimiter and a unix timestamp to pandas daframe as timeseries.
    """
    DF = pd.read_csv(path, delim_whitespace = True)
    DF_ts = UNIXDataFrameColumn2DataIndex(DF, unix_column)
    return DF_ts


def read_batdat(path):
    """
    Read the battery data file in AMPL format and provide the information as dictionary
    """
    bat_filename = '{}/{}'.format(path, bat_dat_filename)
    df = pd.read_csv(bat_filename, sep = ' ''|'';', header = None,
                     skiprows = n_comments(bat_filename, '#'), index_col = 1)
    bat_dict = df.to_dict()[3]
    return bat_dict


def write_spec(name, isfolder, szenname, paramDict = {"Error": "No paramDict available!"}):
    """
    Write specification file by building a dictionary and writing it as columns into a textfile.
    """
    
    spec_list = name.split('_')
    if isfolder:
        filesspec = {'period':  spec_list[-2], 'overlap':  spec_list[-3], 'length':  spec_list[-4],
                     'name': '_'.join(spec_list[:-4]), 'szenname': szenname, 'fileorfolder': 'folder'}
    else:
        filesspec = {
            'name':  name, 'szenname': szenname, 'fileorfolder': 'file'}
    filesspec.update(paramDict)
    w = csv.writer(open(bat_spec_dir, "w"))
    for key, val in filesspec.items():
        w.writerow([key, val])

    print ('\nSpecifications written in output/spec.txt\n')
    return filesspec


def write_period(lastSOC, inputFile, inputFile2 = None):
    """
    Write the period file with the last SoC and the name of the executed file.
    """
    
    with open(period_dat_filename, 'w') as period_file:
        if not inputFile2:
            template = '#This file contains all data adjusted for each optimization step.\nparam sto_soc_start := {SOC};\ndata {input};'
            filled_template = template.format(SOC = lastSOC, input = inputFile)
        else:
            template = '#This file contains all data adjusted for each optimization step.\nparam sto_soc_start := {SOC};\ndata {input};\ndata {inputex};'
            filled_template = template.format(
                SOC = lastSOC, input = inputFile, inputex = inputFile2)
        period_file.write(filled_template)


def getNumberofLastFileinFolder(path):
    """
    Get first element of last (sorted by name) file (not folder) of the name seperated by underscore ("_")
    [1] path of folder
    """
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles = sorted(onlyfiles, key = lambda x: int(x.split("_")[0]))
    filename = onlyfiles.pop()
    return int(filename.split("_")[0])

def parse_generic_output(specDict, step=0):
    # Read AMPL output file and make a table

    column_names = ['Name', 'Eq', 'T', 'Value']

    DF = pd.read_csv(generic_folder, delimiter='\[''|''\,''|'']', header=None, names=column_names)

    DF.iloc[:, 1] = DF.iloc[:, 1].str.replace("'", '')  # replace quote symbol from column 1

    DFindex = DF['Name']

    DFbody = DF[['Eq', 'T', 'Value']]
    DFbodytrans = DFbody.T
    to_shift = pd.isnull(DFbodytrans.iloc[-1])
    DFbodytrans.loc[:, to_shift] = DFbodytrans.loc[:, to_shift].shift(1)
    DFbody2 = DFbodytrans.T

    DFbody2['Name'] = DFindex
    DFbody2.fillna('', inplace=True)

    DFbody2['Name'] = DFbody2['Name'] + DFbody2['Eq']
    del DFbody2['Eq']
    DFbody2 = DFbody2[:-1]

    DFbody2[['T']] = DFbody2[['T']].astype(int)
    DFbody2[['Value']] = DFbody2[['Value']].astype(float)

    table = pd.pivot_table(DFbody2, values='Value', index=['T'], columns=['Name'], aggfunc=np.sum)

    table = table.ix[1:]

    if specDict['fileorfolder'] == 'folder':
        print (specDict)
        orginput_ts = DATAinput2TimeSeries('{0}/{name}/{1}_{name}'.format(input_folder, step, **specDict),
                                           unix_column='UNIX_TIME', escapechar=None, delim_whitespace=True, verbose=1,
                                           cutendlines=2, skipinitialspace=True)
        output_file = '{0}/{1}_res_{name}'.format(parsed_files_dir, step, **specDict)
    else:
        orginput_ts = DATAinput2TimeSeries('{0}/{name}'.format(input_folder, **specDict), unix_column='UNIX_TIME',
                                           escapechar=None, delim_whitespace=True, verbose=1,
                                           cutendlines=2, skipinitialspace=True)
        output_file = '{0}/res_{szenname}_{name}'.format(output_folder, **specDict)

    columnlist = list(orginput_ts.columns.values)

    for item in columnlist:
        if (item != ':=') and (item != 'param:'):
            table[item] = np.array(orginput_ts[item])

    table.to_csv(output_file, sep=" ")
    print ('The parsed file is in: {0}\n'.format(output_file))


def combineResults(folder, period = 'D', length = 1, start = 0, end = 0, unix_column = 'hoy'):
    """
    Combine all the files located in the folder into one single dataframe.
    """
    sorted_file_list = sorted_list_of_files(folder)

    if end != 0:
        sorted_file_list = sorted_file_list[:end]
    if start != 0:
        sorted_file_list = sorted_file_list[start:]

    number_of_files = len(sorted_file_list)
    firstFile = sorted_file_list.pop(0)
    print('\nCombining files\nThe first file in the folder is called: {0}\nThere are {1}files to be combined.'. format(
        firstFile, number_of_files))
    path = '{0}/{1}'.format(folder, firstFile)
    global_df = subPartinDF(path, period, length, unix_column)
    for fileName in sorted_file_list:
        path = '{0}/{1}'.format(folder, fileName)
        next_files_sub_part = subPartinDF(path, period, length, unix_column)
        try:
            global_df = global_df.append(next_files_sub_part, ignore_index = False)
        except AttributeError:
            print("The KeyError caused an AttributeError!")
    return global_df


def loopSubFiles(name, loopStart = 0, loopEnd = None, correct = False, szenname = "", paramDict = None, realDataFile = None,
                 extLoadFile = "",
                 beginSOC = 0.0):
    """
    loops ampl over the files in a folder or just solves a file
    [name] name of the file ormyFolder folder to work with
    [loopStart] first file: 0 is first in folder
    [loopEnd] last file: if None  last in folder is automatically used
    [correct]  1 if predicted is validated with real data. 0 if only predicted data is used
    [szname]  Defined scenario name to keep track of conditions used
    [paramDict] Dictionary. Parameters used in the simulation
    [realDataFile] : File with real PV and Load. Used to calculate real power flows after optimizing with predicted
    values.
    [extLoadFile] File written in perdiod.dat to be read load data file.
    """

    # Starting SOC.

    fileorfolderpath = '{0}/{1}'.format(input_folder, name)
    if paramDict is not None:
        write_spec(name, folder_or_file(fileorfolderpath), szenname, paramDict)
    else:
        write_spec(name, folder_or_file(fileorfolderpath), szenname)

    # System characteristics from the battery
    bat_dict = read_batdat(myFolder)
    PVscale = bat_dict['scale_pv']
    loadscale = bat_dict['scale_load']
    stoCap = bat_dict['sto_cap_max']

    # If correct then read real data from different file
    if correct:
        inputfolder = input_folder  # "input"
        inputpath = '{0}/{1}'.format(inputfolder, realDataFile)
        print('Reading input file: {}\n'.format(inputpath))
        orginput_ts = DATAinput2TimeSeries(inputpath, unix_column = 'hoy', sep = None, escapechar = None,
                                           delim_whitespace = True, verbose = 0, cutendlines = 2)

        cols = ['hoy', 'Load_Pel', 'PV1_Pel']
        orginput_ts = orginput_ts[cols]
        orginput_ts['hoy'] = orginput_ts['hoy'].astype(int)
        orginput_ts = orginput_ts.resample('1min', how = 'mean')

    if folder_or_file(fileorfolderpath):  # if Folder: several files
        delete_files_in_folder(parsed_files_dir)  # 'output/parsed')
        delete_files_in_folder(merged_files_dir)  # 'output/merged')
        specDict = csv_to_dict(bat_spec_dir)
        period = specDict['period']
        length = specDict['length']

        if loopEnd is None:
            print ('{0}/{1}'.format(input_folder, name))

            loopEnd = getNumberofLastFileinFolder(
                '{0}/{1}'.format(input_folder, name))
        for step in range(int(loopStart), int(loopEnd + 1)):           
            if step == loopStart:
                lastSOC = beginSOC
                lastRealSOC = beginSOC
                print('the first step ({0}) of the optimisation starts with a SOC of {1}'.format(
                    loopStart, lastSOC))

            else:
                if not correct:
                    lastFILE = '{0}/{1}_res_{2}'.format(parsed_files_dir, step - 1, name)
                    print (lastFILE)
                    cut_sub = subPartinDF(lastFILE, period, length, 'hoy')
                    lastSOC = float(cut_sub.sto_soc[-1:])
                    if lastSOC > 1:
                        lastSOC = 1
                    if lastSOC < 0:
                        lastSOC = 0

                print('the {0} step of the optimisation starts with a SOC of {1}'.format(
                    step, lastSOC))


            if extLoadFile == '':
                FILE = '{0}/{1}/{2}_{3}'.format(input_folder, name, step, name)
                write_period(lastSOC, FILE)
                call(['ampl', 'simple.run'])
                call(['{0}/auto_generic.sh'.format(output_folder), '{0}/{1}_res_{2}'.format(parsed_files_dir, step, name),
                      '{0}/{1}'.format(generic_folder, 'generic_out.dat'), FILE, str(PVscale), str(loadscale)])
            else:
                FILE = '{0}/{1}/{2}_{3}'.format(input_folder, name, step, name)
                FILEex = '{0}/{1}/{2}_{3}'.format(input_folder,
                                                  extLoadFile, step, extLoadFile)
                write_period(lastSOC, FILE, FILEex)
                call(['ampl', 'simple.run'])
                call(['{0}/auto_generic2.sh'.format(output_folder), '{0}/{1}_res_{2}'.format(parsed_files_dir, step, name),
                      '{0}/{1}'.format(generic_folder, 'generic_out.dat'), FILE, str(PVscale), str(loadscale), FILEex])

            if correct:
                print(
                    'Merging results and input file for the step {0}'.format(step))

                # "output/parsed/", 1)
                listOFfiles = sorted_list_of_files(parsed_files_dir, 1)
                outputpath = '{0}/{1}'.format(parsed_files_dir, listOFfiles[-1])
                outputmerged = merged_files_dir  # "output/merged"
                lastRealCap = lastRealSOC * stoCap
                lastCap = lastSOC * stoCap
                relevantPartofoutput_ts = upsampling(
                    subPartinDF(outputpath, period, length), lastCap)
                relevantPartoforginput_ts = cutsubbyts(
                    orginput_ts, relevantPartofoutput_ts)
                merged_ts = inoutmerge(
                    relevantPartoforginput_ts, relevantPartofoutput_ts, PVscale, loadscale)
                merged_ts = surplus4merged(merged_ts)
                print(
                    'Calculating real battery operation and energy flows for the step {0}\n'.format(step))


# TEST TU ADD BATTERY DISCHARGE
               # merged_ts = surplus2bat_basic(merged_ts)
                merged_ts = surplus2bat(merged_ts, lastRealCap, bat_dict)
#################
                merged_ts = batPVLoad2newbilance(merged_ts, bat_dict)
                merged_ts.to_csv(
                    "{}/{}_merged.dat".format(outputmerged, listOFfiles[-1]), sep = " ")
                mergedFILE = '{0}/{1}_res_{2}_merged.dat'.format(
                    merged_files_dir, step, name)
                cut_sub = subPartinDF(mergedFILE, period, length, 'hoy_x')
# ACTUALIZE SOC WHEN IT IS SMALLER THAN REAL SOC
                lastRealSOC = float(cut_sub.sto_soc_x[-1:])
                lastSOC = float(cut_sub.sto_soc[-1:])
                # Only update lastSoC in case it is below the lastRealSOC. This
                # rule helps to add more "charging allowance" when not much sun
                # is expected.
                if (lastRealSOC > lastSOC):
                    lastSOC = lastRealSOC
                # lastSOC = float(cut_sub.sto_soc_x[-1:])  # This would be the standard form to update the value in model predictive control.
##################
                if lastSOC > 1:
                    lastSOC = 1
                if lastSOC < 0:
                    lastSOC = 0

            if step == int(loopEnd):
                print ('Successfully looped through all files of {0}. Specifications located in output/spec.txt'.format(name))
        global_df = combineResults(parsed_files_dir, period, length)
        final_filename = 'res_{szenname}_{inputfile1}{inputfile2}.dat'.format(
            **specDict)
        global_df.to_csv('{}/{}'.format(output_folder, final_filename), sep =' ')
        print(
            'The combined file is in: {0}/{1}\n'.format("output", final_filename))


        if correct:
            global_df = combineResults(merged_files_dir, period, length, 0, 0, 'hoy_x')
            final_filename = 'res_{szenname}_{inputfile1}{inputfile2}{realdatafile}_merged.dat'.format(
                **specDict)
            global_df.to_csv('{}/{}'.format(output_folder, final_filename), sep =' ')
            print(
                'The combined file is in: {0}/{1}\n'.format("output", final_filename))


    if not folder_or_file(fileorfolderpath):  # if File
        lastSOC = beginSOC
        FILE = '{0}/{1}'.format(input_folder, name)
        write_period(lastSOC, FILE)
        call(['ampl', 'simple.run'])
        newfilename = 'res_{0}_{1}'.format(szenname, name)
        call(['{0}/auto_generic.sh'.format(output_folder), '{0}/{1}'.format(output_folder, newfilename),
              '{0}/{1}'.format(generic_folder, 'generic_out.dat'), FILE, str(PVscale), str(loadscale)])

        if correct:
            print ('Merging results and input file')

            outputpath = '{0}/{1}'.format(output_folder, newfilename)
            lastCap = lastSOC * stoCap
            relevantPartofoutput_ts = upsampling(
                readFile2DFts(outputpath, 'hoy'), lastCap)
            relevantPartoforginput_ts = cutsubbyts(
                orginput_ts, relevantPartofoutput_ts)
            merged_ts = inoutmerge(
                relevantPartoforginput_ts, relevantPartofoutput_ts, PVscale, loadscale)
            merged_ts = surplus4merged(merged_ts)
            print ('Calculating real battery operation and energy flows\n')

# TEST TU ADD BATTERY DISCHARGE
          #  merged_ts = surplus2bat_basic(merged_ts)
            merged_ts = surplus2bat(merged_ts, lastCap, bat_dict)
#################
            merged_ts = batPVLoad2newbilance(merged_ts, stoCap)
            newfilename = newfilename[:-4]
            merged_ts.to_csv(
                "{0}/{1}_merged.dat".format(output_folder, newfilename), sep =" ")

        print(
            'Successfully looped through all files of {0}. Specifications located in output/spec.txt'.format(name))



def bat(i, lastCap, surplus, batMin = 0.0, batMax = 8.0, powerMin = -10.0, powerMax = 10.0, maxpeak = 0.6, period = 1.0 / 60,
        bat_eff = 0.98, inverter_eff = 0.97, bat_self_discharge = 0.07, verbose = 0):
    """
    Calculate sto_in (power +), sto_out (power -) and battery capacity.

    INPUTS
    [1] Starting battery capacity in kWh
    [2] Surplus power in kW. Excess (+) or Deficit (-)
    [3] minimum battery capacity in kWh
    [4] maximum battery capacity in kWh
    [5] maximum power to discharge the battery in kW (negative value)
    [6] maximum power to charge the battery in kW (positive value)
    [7] period in hours
    [bat_eff] : Battery charge/discharge efficiency [Roundtrip eff = 0.96, one-way eff = sqrt(0.96) = 0.97979]
    [inverter_eff] Inverter efficiency
    bas_self_sdicharge : Discharge rate over one month

    RETURN
    [realCap] Final battery capacity in kWh
    [power] Power in kW. Supplied (sto_in) to the battery (positive) or Delivered (sto_out) by the battery (negative)
    """
    bat_dict = read_batdat(myFolder)
    PVscale = bat_dict['scale_pv']
    stoCap = bat_dict['sto_cap_max']

    power = surplus
    if powerMin > power:
        power = powerMin
    if powerMax < power:
        power = powerMax

    bat_charge_discharge_losses = (1 - 0.950789146)  # (1-bat_eff*inverter_eff)
    # Equation to calculate time losses per hour considering 30 days and 24
    # hours
    bat_time_losses = 1 - (1 - bat_self_discharge) ** (period / (30 * 24))

    """
    Traditional battery operation: charge the battery when there is a possitive surplus and discharge when there is a negative surplus
    Battery Capacity = last capacity + energy in/out - charge/discharge
    losses - time losses
    """

# To equal HeiPhoss modell:
    if power >= 0.0:
        theoreticalCap = lastCap / (1 + bat_time_losses) + (
            power * period * (1 - bat_charge_discharge_losses)) / (1 + bat_time_losses)
    else:
        theoreticalCap = lastCap / (1 + bat_time_losses) + (
            power * period / (1 - bat_charge_discharge_losses)) / (1 + bat_time_losses)

    if batMin < theoreticalCap < batMax:
        realCap = theoreticalCap
        power = power

    elif theoreticalCap >= batMax:

        # means it is charging. It is allowed to charge up to batMax
        if lastCap <= batMax:
            realCap = batMax
            power = (realCap * (1 + bat_time_losses) - lastCap) / \
                (period * (1 - bat_charge_discharge_losses))
        # can be charging or discharging but charging is not allowed
        elif lastCap > batMax:
            realCap = theoreticalCap
            power = (realCap * (1 + bat_time_losses) - lastCap) * \
                (1 - bat_charge_discharge_losses) / (period)
            if power < 0:
                power
            else:
                power = 0
        if power > powerMax:
            power = powerMax
        elif power < powerMin:
            power = powerMin

    elif theoreticalCap <= batMin:
        if lastCap >= batMin:
            realCap = batMin
            power = (realCap * (1 + bat_time_losses) - lastCap) * \
                (1 - bat_charge_discharge_losses) / (period)
        elif lastCap < batMin:
            realCap = theoreticalCap
            power = (realCap * (1 + bat_time_losses) - lastCap) / \
                (period * (1 - bat_charge_discharge_losses))
            if power > 0:
                power
            else:
                power = 0
        if power > powerMax:
            power = powerMax
        elif power < powerMin:
            power = powerMin

# to equal HeiPhoSS
    # access this function only if there is a peak level and an excess of
    # power generated
    if (maxpeak < 1.0 and surplus > 0.0):
        # surplus minus power equals feed-in (if possitive). If feed-in is
        # greater than the limit, charge the battery
        if surplus - power > maxpeak * PVscale:
            power = surplus - maxpeak * PVscale
            theoreticalCap = lastCap / (1 + bat_time_losses) + (
                power * period * (1 - bat_charge_discharge_losses)) / (1 + bat_time_losses)
            if theoreticalCap > stoCap:
                realCap = stoCap
                power = (realCap * (1 + bat_time_losses) - lastCap) / \
                    (period * (1 - bat_charge_discharge_losses))
        if power > powerMax:
            power = powerMax
        elif power < powerMin:
            power = powerMin

    # realCap = lastCap + power*period - abs(power)*bat_charge_discharge_losses*period - lastCap*bat_time_losses*period
    if power >= 0.0:
        realCap = lastCap / (1 + bat_time_losses) + (power * period * 
                                                     (1 - bat_charge_discharge_losses)) / (1 + bat_time_losses)
    else:
        realCap = lastCap / (1 + bat_time_losses) + (power * period / 
                                                     (1 - bat_charge_discharge_losses)) / (1 + bat_time_losses)

    dictofbat = {"Cap": realCap, "P": power,
                 "surplus": surplus, "batMAX": batMax, "i": i}
# < #######################################################################

    if verbose == 1:
        with open("log_bat.dat", "a") as f:
            w = csv.writer(f, delimiter = ' ')
            w.writerow(dictofbat.items())
    return dictofbat


def subPartinDF(path, period = 'D', period_numbers = 1, unix_column = 'hoy'):
    """
    Get first period of file by converting it to pandas time series.
    Knows days hours minutes and seconds. Be careful with the timezone, since duplicated timeseries (e.g. summertime
    change) can rise an error.
    """
    period_numbers = float(period_numbers)
    df_ts = readFile2DFts(path, unix_column)
    first_time_step = df_ts.index[0]
    first_time_step = first_time_step.replace(second = 0)
    periodDict = {'D': 'days', 'H': 'hours', 'M': 'minutes', 'S': 'seconds'}
    last_time_step = first_time_step + \
        timedelta(
            **{periodDict[period]: period_numbers}) - timedelta(seconds = 1)
    try:
        subPart = df_ts.ix[str(first_time_step): str(last_time_step)]
    except KeyError:
        ErrorM = 'There was a KeyError in file {0} in the timeperiod between {1} and {2}! Maybe it is caused by day time shift e.g. summer time to winter time and therefore duplicate indexes.'.format(
            path, first_time_step, last_time_step)
        print (ErrorM)
        subPart = ErrorM
    return subPart


def upsampling(ts, lastCap, original_resolution = '15min', new_resolution = '1min'):
    """
    Increase frequency of a pandas timeseries by filling it up from the start
    """
    ts = ts.resample(original_resolution, how = 'mean')
    old_length = len(ts)
    new_length = old_length * \
        (int(original_resolution[:-3]) / int(new_resolution[:-3]))
    index = pd.date_range(ts.index[0], periods = new_length, freq = new_resolution)
    maxnumberoffill = int(original_resolution[:-3]) - int(new_resolution[:-3])
    ts1 = ts.reindex(index, limit = 14, method = 'ffill')
    # function to make an interpolated change over sto_cap instead of 15 minutes steps
    # lastCap is used as initial value but then is eliminated to avoid
    # duplicity of values (between lastSoC and firstSoC)
    index2 = pd.date_range(
        ts.index[0], periods = (new_length + 1), freq = new_resolution)
    ts2 = ts.reindex(index2)
    # shift 15 periods to include the last SoC as value in position 0
    ts2 = ts2.shift(periods = 15)
    ts2['sto_cap'][0] = lastCap  # last SoC is considered the initial value
    ts2['sto_soc'][0] = ts2['sto_soc'][15]
    ts2 = ts2.apply(pd.Series.interpolate)
    # shifted one period to eliminate lastSoC and start with value in position
    # 1
    ts2 = ts2.shift(periods = -1)
    ts2 = ts2.reindex(index)
    # Replace the interpolated sto_cap in the original ts
    ts1['sto_cap'] = ts2['sto_cap']
    ts1['sto_soc'] = ts2['sto_soc']
    return ts1


def cutsubbyts(tobecutted_ts, period_ts):
    starttime = period_ts.index[0]
    endtime = period_ts.index[-1]
    cutted_ts = tobecutted_ts[starttime: endtime]
    return cutted_ts


def inoutmerge(in_ts, out_ts, PVscale = 1, loadscale = 1):
    """
    Scales the PV values from the input, makes both load columns positive and merges in and out in one ts
    """
    
    if out_ts.Load_Pel.min() < 0:
        out_ts['Load_Pel'] = out_ts.Load_Pel * -1 * loadscale
        print ("For the OUTPUT file I multiplied 'Load_Pel' by -1 to change it to positive values.\n")

    else:
        out_ts['Load_Pel'] = out_ts.Load_Pel * loadscale
    if in_ts.Load_Pel.min() < 0:
        in_ts['Load_Pel'] = in_ts.Load_Pel * -1
        print ("For the INPUT file I multiplied 'Load_Pel' by -1 to change it to positive values.\n")

    else:
        in_ts['Load_Pel'] = in_ts.Load_Pel * loadscale

    in_ts['PV1_Pel'] = in_ts.PV1_Pel * PVscale

    merged_ts = pd.merge(in_ts, out_ts, left_index = True, right_index = True)
    merged_ts = merged_ts.tz_convert('Europe/Amsterdam')
    return merged_ts


def surplus4merged(merged_ts):
    """
    Calculate the surplus from input and output by subtracting the electric load from PV production:
    result is positive for surplus and negative for deficit.
    """
    merged_ts['surplus_x'] = merged_ts['PV1_Pel_x'] - merged_ts['Load_Pel_x']
    merged_ts['surplus_y'] = merged_ts['PV1_Pel_y'] - merged_ts['Load_Pel_y']
    return merged_ts


def surplus2bat_basic(merged_ts):
    """
    Use surplus and bat() to calcualte power into and from battery (outer value) as well as the course of the capacity.
    """
    
    firstCap = merged_ts.sto_cap[0]
    lastCap = firstCap
    batPowers = []
    batCaps = []
    for i in range(0, len(merged_ts)):
        batMax = merged_ts.sto_cap[i]
        batDict = bat(i, lastCap, merged_ts.surplus_x[i], 0, batMax)
        batPowers.append(batDict['P'])
        batCaps.append(batDict['Cap'])
        lastCap = batDict['Cap']
    merged_ts['sto_in_out_x'] = batPowers
    merged_ts['sto_cap_x'] = batCaps
    return merged_ts


def surplus2bat(merged_ts, realCap, bat_dict):
    """
    Use surplus and bat() to calcualte power into and from battery (outer value) as well as the course of the capacity.
    """
    
    stoCap = bat_dict['sto_cap_max']
    batMin = bat_dict['sto_min_charge']
    powerMax = bat_dict['sto_max_charge']
    powerMin = -bat_dict['sto_max_discharge']
    maxFeed = bat_dict['feed_in_limit']

    firstCap = merged_ts.sto_cap[0]
    lastCap = firstCap
    lastrealCap = realCap
    batPowers = []
    batCaps = []
    
    for i in range(0, len(merged_ts)):
        # Use the maximum between the (a) planned capacity and (b) desired
        # capacity
        batMax = max(merged_ts.sto_cap[i], merged_ts.sto_soc[i] * stoCap)
        # (a)Planned capacity is usefull to avoid discharging the battery when a last is expected
        # (b)Desired capacity is usefull to increase the amount of charged power when short periods of sun are expected
        # don't discharge battery if there is surplus or battery is under XX
        # kWh of its capacity or realCap is less than desired plan
        if (merged_ts.surplus_x[i] > 0 or lastrealCap < merged_ts.sto_cap[i] or lastrealCap <= 0.8):
            surplusminusbatgrid = merged_ts.surplus_x[i]
        else:
            surplusminusbatgrid = merged_ts.surplus_x[
                i] - merged_ts.P_sto_grid_feed[i]

        batDict = bat(i, lastrealCap, surplusminusbatgrid,
                      batMin, batMax, powerMin, powerMax, maxFeed)
        batPowers.append(batDict['P'])
        batCaps.append(batDict['Cap'])
        lastrealCap = batDict['Cap']
        
    merged_ts['sto_in_out_x'] = batPowers
    merged_ts['sto_cap_x'] = batCaps
    return merged_ts


def batPVLoad2newbilance(merged_ts, bat_dict):
    """
    Calculate the parameter values from the AMPL output to compare
    by numpy array calculations and add them with a ts in the end to
    the pandas dataframe.
    """

    stoCap = bat_dict['sto_cap_max']
    maxFeed = bat_dict['feed_in_limit']
    PVscale = bat_dict['scale_pv']
    loadscale = bat_dict['scale_load']

    merged_ts['PV_local_x'] = np.where(merged_ts.Load_Pel_x + merged_ts.sto_in_out_x >= merged_ts.PV1_Pel_x,
                                       merged_ts.PV1_Pel_x,
                                       np.where(merged_ts.Load_Pel_x + merged_ts.sto_in_out_x > 0,
                                                merged_ts.Load_Pel_x + 
                                                merged_ts.sto_in_out_x,
                                                0))

    merged_ts['PV_self_x'] = np.where(merged_ts.PV1_Pel_x - merged_ts.sto_in_out_x > merged_ts.Load_Pel_x,
                                      merged_ts.Load_Pel_x,
                                      np.where(merged_ts.PV1_Pel_x - merged_ts.sto_in_out_x > 0,
                                               merged_ts.PV1_Pel_x - 
                                               merged_ts.sto_in_out_x,
                                               0))

    merged_ts['PV_direct_x'] = np.where(merged_ts.PV_local_x - merged_ts.PV_self_x >= 0,
                                        merged_ts.PV_self_x,
                                        merged_ts.PV_local_x)

    merged_ts['PV_grid_feed_x'] = np.where(merged_ts.PV1_Pel_x - merged_ts.PV_local_x <= PVscale * maxFeed,
                                           merged_ts.PV1_Pel_x - 
                                           merged_ts.PV_local_x,
                                           PVscale * maxFeed)

    merged_ts['Feed_limit_losses_x'] = np.where(merged_ts.PV1_Pel_x - merged_ts.PV_local_x > PVscale * maxFeed,
                                                merged_ts.PV1_Pel_x - 
                                                merged_ts.PV_local_x - 
                                                PVscale * maxFeed,
                                                0)

    merged_ts['grid_load_supply_x'] = merged_ts.Load_Pel_x - \
        merged_ts.PV_self_x

    merged_ts['sto_soc_x'] = merged_ts.sto_cap_x / stoCap

    merged_ts['sto_grid_discharge_x'] = np.where(merged_ts.sto_in_out_x < 0,
                                                 - merged_ts.sto_in_out_x - 
                                                 (merged_ts.Load_Pel_x - merged_ts.PV_direct_x - 
                                                  merged_ts.grid_load_supply_x),
                                                 0)

    return merged_ts


# CORE
if __name__ == "__main__":


    from sys import argv
    print(len(sys.argv))
    print(sys.argv[0])
    loopSubFiles(str(argv[1]), int(argv[2]), int(
        argv[3]), int(argv[4]), int(argv[5]))
