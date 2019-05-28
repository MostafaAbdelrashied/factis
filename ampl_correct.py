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

import csv
from subprocess import call
import pandas as pd
import numpy as np
from datetime import timedelta
from os import listdir
from os.path import isfile, join
import getpass
import datetime

# + local/relativ modul import
import inspect
import os

from paths import *

# genpy imports
from genpy.datacrunching import UNIXDataFrameColumn2DataIndex, DATAinput2TimeSeries, csv_to_dict, n_comments
from genpy.systools import sorted_list_of_files, delete_files_in_folder, folder_or_file
from plot_framework import readamplres
# -

myFolder = (os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe()))))
# Functions


def readFile2DFts(path, unix_column='hoy'):
    """
    Reads CSV with whitespace delimiter and a unix timestamp to pandas daframe as timeseries.
    """
    DF = pd.read_csv(path, delim_whitespace=True)
    DF_ts = UNIXDataFrameColumn2DataIndex(DF, unix_column)
    return DF_ts


def read_batdat(path):
    """
    Read the battery data file in AMPL format and provide the information as dictionary
    """
    bat_filename = '{}/{}'.format(path, bat_dat_filename)
    df = pd.read_csv(bat_filename, sep=' ''|'';', header=None,
                     skiprows=n_comments(bat_filename, '#'), index_col=1, engine='python')
    bat_dict = df.to_dict()[3]
    return bat_dict


def write_spec(name, isfolder, szenname, paramDict={"Error": "No paramDict available!"}, price_col=None, simplerun='simple.run'):
    """
    Write specification file by building a dictionary and writing it as columns into a textfile.
    """

    spec_list = name.split('_')
    if isfolder:
        filesspec = {'period': spec_list[-2], 'overlap': spec_list[-3], 'length': spec_list[-4],
                     'name': name, 'szenname': szenname, 'fileorfolder': 'folder', 'price_col': price_col, 'AMPL_run_file': simplerun}
    else:
        filesspec = {
            'name': name, 'szenname': szenname, 'fileorfolder': 'file'}
    filesspec.update(paramDict)
    w = csv.writer(open(bat_spec_dir, "w", newline="\n", encoding="utf-8"))
    for key, val in filesspec.items():
        w.writerow([key, val])

    print('\nSpecifications written in output/spec.txt\n')
    return filesspec


def write_period(lastSOC, inputFile, inputFile2=None):
    """
    Write the period file with the last SoC and the name of the executed file.
    """

    with open(period_dat_filename, 'w') as period_file:
        if not inputFile2:
            template = '#This file contains all data adjusted for each optimization step.\nparam sto_soc_start := {SOC};\ndata {input};'
            filled_template = template.format(SOC=lastSOC, input=inputFile)

            filled_template = template.format(SOC=lastSOC, input=str(inputFile).replace('\\', '/'))

        else:
            template = '#This file contains all data adjusted for each optimization step.\nparam sto_soc_start := {SOC};\ndata {input};\ndata {inputex};'
            filled_template = template.format(
                SOC=lastSOC, input=inputFile, inputex=inputFile2)

        period_file.write(filled_template)


def resample_res_output(df, new_res='15min'):
    df = charge_duplicate_remove(df, discharge_col='sto_discharge', ref_col='bat_change')
    df = df.resample(new_res).ffill()
    df.insert(0, 'T', range(1, len(df) + 1))
    df['hoy'] = df.index.astype(np.int64) // 10**9
    return df


def fix_parameter(df, date=None, df_cols=['bat_change', 'profit',
                                          'sto_soc', 'sto_cap', 'sto_in', 'sto_out',
                                          'sto_charge', 'sto_discharge', 'bat_in', 'bat_out',
                                          'Day_Ahead_Auction'], fname='fix_01.dat'):

    if date:
        temp_df = df.loc[date, df_cols]
    else:
        temp_df = df.loc[:, df_cols]

    df2 = temp_df.reset_index()
    file = open('correct/{}'.format(fname), 'w')
    for column in df_cols:
        for time, value in df2[column].items():
            file.write('fix {}_o [{}] := {};\n'.format(column, time + 1, value))
    print('Done writing {} file'.format(fname))


def daily_profit(df, price_col, rt_eff):
    """
    Calculating the revenu, cost and profit for each day and append it to the result file
    """
    step = df.index.to_series().diff().dt.seconds.div(3600, fill_value=0)[1]
    df['profit_c'] = df[price_col] * df['bat_change_c'] * step * rt_eff**0.5 / 1000
    df['profit_o'] = df['Day_Ahead_Auction_o'] * df['bat_change_o'] * step * rt_eff**0.5 / 1000
    df['profit'] = df['profit_o'] + df['profit_c']
    # check
    total_profit = df['profit'].sum()
    days_no = len(df.resample('D'))
    print('The total optimized profit for these {0:d} days = {1:f} Euro'.format(days_no, total_profit))


def write_profit(step, name):
    """
    Extract the optimized profit value from the log file and document it down in the profit log file
    """
    with open(log_dat_path_server, 'r') as log_file:
        for line in log_file:
            s = line.split(' ')
            for i, j in enumerate(s):
                if j == "objective":
                    name, value = str(s[i]), float(s[i + 1])
                    print('{0:s} profits = {1:f} Euro\n'.format(name, value))
    open(profit_log, 'a').write('The file {0}_{1} Optimized Profit = {2:f} Euro\n'.format(step, name, value))


def profit_sum():
    """
    Summing up the total profit and append it to the end of the profit log file
    """
    profit_value = []
    with open(profit_log, 'r') as prof_file:
        for line in prof_file:
            s = line.split(' ')
            for i, j in enumerate(s):
                if j == "Profit":
                    value = float(s[i + 2])
                    profit_value.append(value)
    profit_sum = sum(profit_value)
    today_time = datetime.datetime.now().isoformat(timespec='seconds')
    open(profit_log, 'a').write('The total profit from the first to last files = {0:f} Euro\n###Created on {1}\n\n\n'.format(profit_sum, today_time))
    print('The total profit from the first to last files = {0:f} Euro\n'.format(profit_sum))


def getNumberofLastFileinFolder(path):
    """
    Get first element of last (sorted by name) file (not folder) of the name seperated by underscore ("_")
    [1] path of folder
    """
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles = sorted(onlyfiles, key=lambda x: int(x.split("_")[0]))
    filename = onlyfiles.pop()
    return int(filename.split("_")[0])


def parse_generic_output(specDict, step=0):
    # Read AMPL output file and make a table

    column_names = ['Name', 'T', 'Value']
    DF = pd.read_csv(out_dat_path_server, delimiter='\[''|''\,''|'']', header=None, names=column_names, engine='python')
    DF.iloc[:, 1] = DF.iloc[:, 1].replace("'", '')  # replace quote symbol from column 1

    DFindex = DF['Name']
    DFbody = DF[['T', 'Value']]
    DFbodytrans = DFbody.T
    to_shift = pd.isnull(DFbodytrans.iloc[-1])
    DFbodytrans.loc[:, to_shift] = DFbodytrans.loc[:, to_shift].shift(1)
    DFbody2 = DFbodytrans.T

    DFbody2['Name'] = DFindex
    DFbody2.fillna('', inplace=True)

    DFbody2 = DFbody2[:-1]

    table = pd.pivot_table(DFbody2, values='Value', index=['T'], columns=['Name'], aggfunc=np.sum)
    table = table.ix[1:]

    if specDict['fileorfolder'] == 'folder':
        orginput_ts = DATAinput2TimeSeries('{0}/{name}/{1}_{name}'.format(input_folder, step, **specDict),
                                           unix_column='hoy', escapechar=None, delim_whitespace=True, verbose=0,
                                           cutendlines=2, skipinitialspace=True)
        output_file = '{0}/{1}_res_{name}'.format(parsed_files_dir, step, **specDict)
    else:
        orginput_ts = DATAinput2TimeSeries('{0}/{name}'.format(input_folder, **specDict), unix_column='hoy',
                                           escapechar=None, delim_whitespace=True, verbose=0,
                                           cutendlines=2, skipinitialspace=True)
        output_file = '{0}/res_{szenname}_{name}'.format(parsed_files_dir, **specDict)

    columnlist = list(orginput_ts.columns.values)
    print(table.shape)
    print(orginput_ts.shape)
    for item in columnlist:
        if (item != ':=') and (item != 'param:'):
            table[item] = np.array(orginput_ts[item])
            orginput_ts[item]

    df_out = pd.DataFrame(table.to_records())
    df_out = df_out.apply(pd.to_numeric, errors='ignore')
    df_out.to_csv(output_file, sep=" ")

    print('The parsed file is in: {0}\n'.format(output_file))


def combineResults(folder, period='D', length=1, start=0, end=0, unix_column='hoy'):
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
    print('\nCombining files\nThe first file in the folder is called: {0}\nThere are {1} files to be combined.'. format(
        firstFile, number_of_files))
    path = '{0}/{1}'.format(folder, firstFile)
    global_df = subPartinDF(path, period, length, unix_column)

    for fileName in sorted_file_list:
        path = '{0}/{1}'.format(folder, fileName)
        next_files_sub_part = subPartinDF(path, period, length, unix_column)
        try:
            global_df = global_df.append(
                next_files_sub_part, ignore_index=False)
        except AttributeError:
            print("The KeyError caused an AttributeError!")

    return global_df


def loopSubFiles(name, loopStart=0, loopEnd=None, correct=False, szenname="", paramDict=None, realDataFile=None,
                 extLoadFile="",
                 beginSOC=0.0):
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
        inputfolder = path(input_folder)  # "input"
        inputpath = '{0}/{1}'.format(inputfolder, realDataFile)
        print('Reading input file: {}\n'.format(inputpath))
        orginput_ts = DATAinput2TimeSeries(inputpath, unix_column='hoy', sep=None, escapechar=None,
                                           delim_whitespace=True, verbose=0, cutendlines=2)

        cols = ['hoy', 'Load_Pel', 'PV1_Pel']
        orginput_ts = orginput_ts[cols]
        orginput_ts['hoy'] = orginput_ts['hoy'].astype(int)
        orginput_ts = orginput_ts.resample('1min', how='mean')

    if folder_or_file(fileorfolderpath):  # if Folder: several files
        delete_files_in_folder(parsed_files_dir)  # 'output/parsed')
        delete_files_in_folder(merged_files_dir)  # 'output/merged')
        specDict = csv_to_dict(bat_spec_dir)
        period = specDict['period']
        length = specDict['length']

        if loopEnd is None:
            print('{0}/{1}'.format(input_folder, name))

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
                    print(lastFILE)
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
                parse_generic_output(specDict, step)
#                 call(['{0}/auto_generic.sh'.format(output_folder), '{0}/{1}_res_{2}'.format(parsed_files_dir, step, name),
#                       '{0}/{1}'.format(generic_folder, 'generic_out.dat'), FILE, str(PVscale), str(loadscale)])
            else:
                FILE = '{0}/{1}/{2}_{3}'.format(input_folder, name, step, name)
                FILEex = '{0}/{1}/{2}_{3}'.format(input_folder, extLoadFile, step, extLoadFile)
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
                    "{}/{}_merged.dat".format(outputmerged, listOFfiles[-1]), sep=" ")
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
                print('Successfully looped through all files of {0}. Specifications located in output/spec.txt'.format(

                    name))
        global_df = combineResults(parsed_files_dir, period, length)
        final_filename = 'res_{szenname}_{inputfile1}{inputfile2}.dat'.format(
            **specDict)
        global_df.to_csv('{}/{}'.format(output_folder, final_filename), sep=',')
        print(
            'The combined file is in: {0}/{1}\n'.format("output", final_filename))

        if correct:
            global_df = combineResults(merged_files_dir, period, length, 0, 0, 'hoy_x')
            final_filename = 'res_{szenname}_{inputfile1}{inputfile2}{realdatafile}_merged.dat'.format(
                **specDict)
            global_df.to_csv('{}/{}'.format(output_folder, final_filename), sep=',')
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
            print('Merging results and input file')

            outputpath = '{0}/{1}'.format(output_folder, newfilename)
            lastCap = lastSOC * stoCap
            relevantPartofoutput_ts = upsampling(readFile2DFts(outputpath, 'hoy'), lastCap)
            orginput_ts = upsampling(readFile2DFts(orginput_ts, 'hoy'), lastCap)
            relevantPartoforginput_ts = cutsubbyts(orginput_ts, relevantPartofoutput_ts)
            merged_ts = inoutmerge(relevantPartoforginput_ts, relevantPartofoutput_ts, PVscale, loadscale)
            merged_ts = surplus4merged(merged_ts)
            print('Calculating real battery operation and energy flows\n')

# TEST TU ADD BATTERY DISCHARGE
            #  merged_ts = surplus2bat_basic(merged_ts)
            merged_ts = surplus2bat(merged_ts, lastCap, bat_dict)
#################
            merged_ts = batPVLoad2newbilance(merged_ts, stoCap)
            newfilename = newfilename[:-4]
            merged_ts.to_csv(
                "{0}/{1}_merged.dat".format(output_folder, newfilename), sep=" ")

        print(
            'Successfully looped through all files of {0}. Specifications located in output/spec.txt'.format(name))


def write_rsyncTo_sx401(inputfolder):

    _user = getpass.getuser()

    rsyncfile = 'rsyncTo_sx401.sh'

    template = ('#/bin/bash \n\n' +
                'rsync -v -r -t -u simple.run sx401:/net/home/{}/{}/amplopt \n'.format(_user[0], _user) +
                'rsync -v -r -t -u bat.dat sx401:/net/home/{}/{}/amplopt \n'.format(_user[0], _user) +
                'rsync -v -r -t -u period.dat sx401:/net/home/{}/{}/amplopt \n'.format(_user[0], _user) +
                'rsync -v -r -t -u {} sx401:/net/home/{}/{}/amplopt/{} \n'.format(inputfolder, _user[0], _user, input_folder))

    with open(rsyncfile, 'w') as rsync_file:
        rsync_file.write(template)


def loopSubFiles_sx401(name, loopStart=0, loopEnd=0, correct=False, szenname="", paramDict=None, realDataFile=None, price_col='Day_Ahead_Auction',
                       extLoadFile="", beginSOC=0.0, ref_file=None, simplerun='simple.run'):
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
    # realdatfile = '{0}/{1}/{2}_{3}'.format(input_folder, realDataFile, step, realDataFile)

#     inputpath = '{0}/{1}'.format(input_folder, realDataFile)
#     print('Reading input file: {}\n'.format(inputpath))
#     orginput_ts = readamplres(inputpath, unix_column='hoy')
#     print('Reading input file: {}\n'.format(inputpath))
    #orginput_ts.insert(1, 'hoy_x', orginput_ts['hoy'].astype(int))
    #orginput_ts.insert(1, 'profit_x', orginput_ts['profit'].astype(int))
    #orginput_ts['hoy_x'] = orginput_ts['hoy'].astype(int)
    #orginput_ts.drop(labels=['hoy'], axis=1, inplace=True)
    fileorfolderpath = '{0}/{1}'.format(input_folder, name)

    if paramDict is not None:
        write_spec(name, folder_or_file(fileorfolderpath), szenname, paramDict, price_col=price_col, simplerun=simplerun)
    else:
        write_spec(name, folder_or_file(fileorfolderpath), szenname)

    # System characteristics from the battery
    bat_dict = read_batdat(myFolder)
    PVscale = bat_dict['scale_pv']
    loadscale = bat_dict['scale_load']
    stoCap = bat_dict['sto_cap_max']

    # Create rsync files
    write_rsyncTo_sx401(fileorfolderpath)

    # If correct then read real data from different file

    #orginput_ts = orginput_ts.resample('15min').mean()

    if folder_or_file(fileorfolderpath):  # if Folder: several files
        delete_files_in_folder(parsed_files_dir)  # 'output/parsed')
        delete_files_in_folder(merged_files_dir)  # 'output/merged')
        specDict = csv_to_dict(bat_spec_dir)
        period = specDict['period']
        length = specDict['length']

        if loopEnd == 0:
            print('{0}/{1}'.format(input_folder, name))

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
                    #astSOC = float(cut_sub.sto_soc[-1:])

                    if lastSOC > 1:
                        lastSOC = 1
                    if lastSOC < 0:
                        lastSOC = 0

                print('the {0} step of the optimisation starts with a SOC of {1}'.format(step, lastSOC))
            if correct:
                pass

            if extLoadFile == '':
                FILE = '{0}/{1}/{2}_{3}'.format(input_folder, name, step, name)
                write_period(lastSOC, FILE)

                if os.name == 'nt':  # For Windows
                    # Copying input.dat files from local computer to sx401
                    # call([rsync -r -v -t -u input_data_path_local input_data_path_server])
                    #call([windows_powershell_path, 'cp -r',input_data_path_local, input_data_path_server])
                    # Copying simple.run, bat.dat, period.dat and model.mod files from local computer to sx401
                    ref_folder = '/ref_files'
                    orginput_path = '{0}/{1}'.format(output_folder + ref_folder, ref_file)
                    orginput_ts = cutsubfile(orginput_path, FILE)
                    # fix_parameter(orginput_ts, df_cols=['bat_change', 'Day_Ahead_Auction', 'sto_cap'], fname='fix_01.dat')

                    fix_parameter(orginput_ts, df_cols=['bat_change_o', 'Day_Ahead_Auction_o', 'sto_cap_o',
                                                        'bat_change_c', 'Intraday_Auction_15_minute_call', 'sto_cap_c'], fname='fix_01.dat')  # df_cols=['bat_change']
                    call([windows_powershell_path, 'cp', bat_dat_path, opt_server_path])
                    call([windows_powershell_path, 'cp', per_dat_path, opt_server_path])
                    call([windows_powershell_path, 'cp', model_filename, model_filename_server])
                    call([windows_powershell_path, 'cp', simple_run_path_corr, opt_server_path])
                    call([windows_powershell_path, 'cp', fix_file, fix_file_path_server])
                    call([windows_powershell_path, 'cp', model_folder, opt_server_path])
#                     if correct:
                    #call([windows_powershell_path, 'cp -r', input_real_data_path_local, input_real_data_path_server])
#                     else:
                    print("Done copying to sx401")
                    # Run AMPL on the server
                    username_server = getpass.getuser() + '@sx401'
                    call([windows_powershell_path, 'ssh', username_server, '"', 'cd', 'amplopt', ';', 'ampl', 'simple_correct.run', '"'])
#                     if correct:
#                     else:
#                         call([windows_powershell_path, 'ssh', username_server, '"', 'cd', 'amplopt', ';', 'ampl', 'simple.run', '"'])

                    parse_generic_output(specDict, step)
                    #write_profit(step, name)
                    print("Done computing on sx401")
                    # Copying  generic_output.dat file from sx401 to local computer
                    call([windows_powershell_path, 'cp', out_dat_path_server, out_dat_path_local])
                    call([windows_powershell_path, 'cp', log_dat_path_server, log_dat_path_local])
                    print("Done copying from sx401")

                else:  # For Linux
                    pass

                # TODO:
                # MAYBE ALSO COPY log_file.txt FROM SERVER?

            else:
                FILE = '{0}/{1}/{2}_{3}'.format(input_folder, name, step, name)
                FILEex = '{0}/{1}/{2}_{3}'.format(input_folder, extLoadFile, step, extLoadFile)
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

                relevantPartofoutput_ts = upsampling(subPartinDF(outputpath, period, length), lastCap, original_resolution='15min', new_resolution='15min')
                relevantPartoforginput_ts = cutsubbyts(orginput_ts, relevantPartofoutput_ts)
                relevantPartoforginput_ts = upsampling(relevantPartoforginput_ts, lastRealCap, original_resolution='60min', new_resolution='15min')

                merged_ts = inoutmerge(relevantPartoforginput_ts, relevantPartofoutput_ts, PVscale, loadscale, price_col)
                merged_ts = surplus4merged(merged_ts)
                merged_ts = charge_duplicate_remove(merged_ts, discharge_col='sto_discharge_x', ref_col='bat_change_x')
                merged_ts = charge_duplicate_remove(merged_ts, discharge_col='sto_discharge_y', ref_col='bat_change_y')
                print(
                    'Calculating real battery operation and energy flows for the step {0}\n'.format(step))


# TEST TU ADD BATTERY DISCHARGE
                # merged_ts = surplus2bat_basic(merged_ts)
                # print(merged_ts.head(10))
                #print(merged_ts.loc[:, ['bat_change_x', 'bat_change_y']])
                # exit()
                merged_ts = surplus2bat(merged_ts, lastRealCap, bat_dict)

                #merged_ts = batPVLoad2newbilance(merged_ts, bat_dict)
                merged_ts.to_csv("{}/{}_merged.dat".format(outputmerged, listOFfiles[-1]), sep=" ", index=False)
                mergedFILE = '{0}/{1}_res_{2}_merged.dat'.format(merged_files_dir, step, name)
                cut_sub = subPartinDF(mergedFILE, period, length, 'hoy_x')
# ACTUALIZE SOC WHEN IT IS SMALLER THAN REAL SOC
                lastRealSOC = float(cut_sub['sto_soc_x'][-1:])
                lastSOC = float(cut_sub['sto_soc_y'][-1:])
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
                print('Successfully looped through all files of {0}. Specifications located in output/spec.txt'.format(
                    name))
        # profit_sum()
        global_df = combineResults(parsed_files_dir, period, length)
        rt_eff = (1 - float(specDict['sto_loss_charge'])) * (1 - float(specDict['sto_loss_discharge']))
        # daily_profit(global_df, price_col, rt_eff)  # float(specDict['szenname'].split('_')[-1]
        final_filename = 'res_{szenname}_{inputfile1}{inputfile2}_{price_col}.dat'.format(**specDict)
        global_df.to_csv('{}/{}'.format(output_folder, final_filename), sep=',')
        # fix_parameter(global_df)
        print('The combined file is in: {0}/{1}\n'.format("output", final_filename))

        if correct:
            global_df = combineResults(merged_files_dir, period, length, 0, 0, 'hoy_x')
            final_filename = 'res_{szenname}_{inputfile1}{inputfile2}{realdatafile}_merged.dat'.format(
                **specDict)
            global_df.to_csv('{}/{}'.format(output_folder, final_filename), sep=',')
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
            print('Merging results and input file')

            outputpath = '{0}/{1}'.format(output_folder, newfilename)
            lastCap = lastSOC * stoCap
            relevantPartofoutput_ts = upsampling(
                readFile2DFts(outputpath, 'hoy'), lastCap)
            relevantPartoforginput_ts = cutsubbyts(
                orginput_ts, relevantPartofoutput_ts)
            merged_ts = inoutmerge(
                relevantPartoforginput_ts, relevantPartofoutput_ts, PVscale, loadscale)
            merged_ts = surplus4merged(merged_ts)
            print('Calculating real battery operation and energy flows\n')

# TEST TU ADD BATTERY DISCHARGE
            #  merged_ts = surplus2bat_basic(merged_ts)
            merged_ts = surplus2bat(merged_ts, lastCap, bat_dict)
#################
            merged_ts = batPVLoad2newbilance(merged_ts, stoCap)
            newfilename = newfilename[:-4]
            merged_ts.to_csv(
                "{0}/{1}_merged.dat".format(output_folder, newfilename), sep=",")

        print(
            'Successfully looped through all files of {0}. Specifications located in output/spec.txt'.format(name))


def bat(i, lastCap, surplus, batMin=0.0, batMax=8.0, powerMin=-10.0, powerMax=10.0, maxpeak=0.6, period=1.0 / 60,
        bat_eff=0.98, inverter_eff=0.97, bat_self_discharge=0.07, verbose=0):
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
            w = csv.writer(f, delimiter=' ')
            w.writerow(dictofbat.items())
    return dictofbat


def subPartinDF(path, period='D', period_numbers=1, unix_column='hoy'):
    """
    Get first period of file by converting it to pandas time series.
    Knows days hours minutes and seconds. Be careful with the timezone, since duplicated timeseries (e.g. summertime
    change) can rise an error.
    """
    period_numbers = float(period_numbers)
    df_ts = readFile2DFts(path, unix_column)
    first_time_step = df_ts.index[0]
    first_time_step = first_time_step.replace(second=0)
    periodDict = {'D': 'days', 'H': 'hours', 'M': 'minutes', 'S': 'seconds'}
    last_time_step = first_time_step + \
        timedelta(
            **{periodDict[period]: period_numbers}) - timedelta(seconds=1)
    try:
        subPart = df_ts.ix[str(first_time_step): str(last_time_step)]
    except KeyError:
        ErrorM = 'There was a KeyError in file {0} in the timeperiod between {1} and {2}! Maybe it is caused by day time shift e.g. summer time to winter time and therefore duplicate indexes.'.format(
            path, first_time_step, last_time_step)
        print(ErrorM)
        subPart = ErrorM
    return subPart


def upsampling(ts, lastCap, original_resolution='60min', new_resolution='15min'):
    """
    Increase frequency of a pandas timeseries by filling it up from the start
    """
    ts = ts.resample(original_resolution).mean()
    old_length = len(ts)
    new_length = old_length * \
        (int(original_resolution[:-3]) / int(new_resolution[:-3]))
    index = pd.date_range(ts.index[0], periods=new_length, freq=new_resolution)
    #maxnumberoffill = int(original_resolution[:-3]) - int(new_resolution[:-3])
    ts1 = ts.reindex(index, limit=14, method='ffill')
    # function to make an interpolated change over sto_cap instead of 15 minutes steps
    # lastCap is used as initial value but then is eliminated to avoid
    # duplicity of values (between lastSoC and firstSoC)
    index2 = pd.date_range(
        ts.index[0], periods=(new_length + 1), freq=new_resolution)
    ts2 = ts.reindex(index2)
    # shift 15 periods to include the last SoC as value in position 0
    ts2 = ts2.shift(periods=15)
    ts2['sto_cap'][0] = lastCap  # last SoC is considered the initial value
    ts2['sto_soc'][0] = ts2['sto_soc'][15]
    ts2 = ts2.apply(pd.Series.interpolate)
    # shifted one period to eliminate lastSoC and start with value in position
    # 1
    ts2 = ts2.shift(periods=-1)
    ts2 = ts2.reindex(index)
    # Replace the interpolated sto_cap in the original ts
    ts1['sto_cap'] = ts2['sto_cap']
    ts1['sto_soc'] = ts2['sto_soc']
    return ts1


def cutsubfile(tobecutted_fname, period_fname):
    """
    Cut the the relevant part from the output file to be simulated accordingly with the revelant part from the input
    """
    period_ts = pd.read_csv(period_fname, sep=' ', skipfooter=2, engine='python', usecols=['hoy'])
    period_ts.set_index('hoy', drop=False, inplace=True)
    tobecutted_ts = readamplres(tobecutted_fname, unix_column='hoy',
                                drop_zeros=False,
                                drop_str=False)
    period_ts_res = UNIXDataFrameColumn2DataIndex(period_ts, unix_column='hoy').index.to_series().diff().dt.seconds.div(60, fill_value=0)[1]
    tobecutted_ts_res = tobecutted_ts.index.to_series().diff().dt.seconds.div(60, fill_value=0)[1]
    if period_ts_res != tobecutted_ts_res:
        tobecutted_ts = resample_res_output(tobecutted_ts, new_res=str(int(period_ts_res)) + 'min')
        print('resampling the reference market to match the objective market has been done successfully with new time_res {}min !!!'.format(period_ts_res))
    period_ts.set_index('hoy', drop=False, inplace=True)
    cutted_ts = tobecutted_ts.loc[tobecutted_ts['hoy'].isin(period_ts.index)]
    return cutted_ts


def cutsubbyts(tobecutted_ts, period_ts):
    """
    Cut the the relevant part from the output file to be simulated accordingly with the revelant part from the input
    """
    starttime = period_ts.index[0]
    endtime = period_ts.index[-1]
    cutted_ts = tobecutted_ts[starttime: endtime]
    return cutted_ts


def inoutmerge(in_ts, out_ts, PVscale=1, loadscale=1, price_col='Day_Ahead_Auction'):
    """
    Scales the PV values from the input, makes both load columns positive and merges in and out in one ts
    """
#     if out_ts.Load_Pel.min() < 0:
#         out_ts['Load_Pel'] = out_ts.Load_Pel * -1 * loadscale
#         print("For the OUTPUT file I multiplied 'Load_Pel' by -1 to change it to positive values.\n")
#
#     else:
#         out_ts['Load_Pel'] = out_ts.Load_Pel * loadscale
#
#
#     if in_ts.Load_Pel.min() < 0:
#         in_ts['Load_Pel'] = in_ts.Load_Pel * -1
#         print("For the INPUT file I multiplied 'Load_Pel' by -1 to change it to positive values.\n")
#
#     else:
#         in_ts['Load_Pel'] = in_ts.Load_Pel * loadscale
#
#     in_ts['PV1_Pel'] = in_ts.PV1_Pel * PVscale
    out_ts['profit'] = round(out_ts[price_col] * (out_ts['sto_out'] - out_ts['sto_in']) / 1000, 2)
    merged_ts = pd.merge(in_ts, out_ts, left_index=True, right_index=True)
    merged_ts = merged_ts.tz_convert('Europe/Amsterdam')
    return merged_ts


def surplus4merged(merged_ts):
    """
    Calculate the surplus from input and output by subtracting the electric load from PV production:
    result is positive for surplus and negative for deficit.
    """
#     merged_ts['surplus_x'] = merged_ts['PV1_Pel_x'] - merged_ts['Load_Pel_x']
#     merged_ts['surplus_y'] = merged_ts['PV1_Pel_y'] - merged_ts['Load_Pel_y']
    return merged_ts


def charge_duplicate_remove(df_temp, discharge_col='sto_discharge', ref_col='bat_change'):
    for index, rows in df_temp.iterrows():
        if df_temp.loc[index, ref_col] == 0:
            df_temp.loc[index, discharge_col] = 0

    return df_temp


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

    lastrealCap = realCap
    sto_caps = []
    sto_ins = []
    bat_ins = []
    sto_outs = []
    bat_outs = []
    profits = []

    for i in range(0, len(merged_ts)):
        # Use the maximum between the (a) planned capacity and (b) desired capacity
        # (a)Planned capacity is usefull to avoid discharging the battery when a last is expected
        # (b)Desired capacity is usefull to increase the amount of charged power when short periods of sun are expected
        # don't discharge battery if there is surplus or battery is under XX
        # kWh of its capacity or realCap is less than desired plan
        if (merged_ts.profit_y[i] > merged_ts.profit_x[i] and merged_ts.profit_x[i] != 0):
            # if merged_ts.sto_cap_x[i] + merged_ts.sto_cap_y[i] < stoCap:
            sto_cap = abs(merged_ts.sto_cap_x[i] - merged_ts.sto_cap_y[i])
            # if merged_ts.sto_in_x[i] + merged_ts.sto_in_y[i] < powerMax:
            sto_in = abs(merged_ts.sto_in_x[i] - merged_ts.sto_in_y[i])
            bat_in = abs(merged_ts.bat_in_x[i] - merged_ts.bat_in_y[i])
            # if merged_ts.sto_out_x[i] + merged_ts.sto_out_y[i] < powerMin:
            sto_out = abs(merged_ts.sto_out_x[i] - merged_ts.sto_out_y[i])
            bat_out = abs(merged_ts.bat_out_x[i] - merged_ts.bat_out_y[i])
            profit = merged_ts.profit_x[i] + merged_ts.profit_y[i]

        elif (merged_ts.profit_y[i] > merged_ts.profit_x[i] or merged_ts.profit_x[i] == 0):
            sto_cap = merged_ts.sto_cap_y[i]
            sto_in = merged_ts.sto_in_y[i]
            bat_in = merged_ts.bat_in_y[i]
            sto_out = merged_ts.sto_out_y[i]
            bat_out = merged_ts.bat_out_y[i]
            profit = merged_ts.profit_y[i]
        else:
            sto_cap = merged_ts.sto_cap_x[i]
            sto_in = merged_ts.sto_in_x[i]
            bat_in = merged_ts.bat_in_x[i]
            sto_out = merged_ts.sto_out_x[i]
            bat_out = merged_ts.bat_out_x[i]
            profit = merged_ts.profit_x[i]

        sto_caps.append(sto_cap)
        sto_ins.append(sto_in)
        bat_ins.append(bat_in)
        sto_outs.append(sto_out)
        bat_outs.append(bat_out)
        profits.append(profit)

    merged_ts['sto_cap'] = sto_caps
    merged_ts['sto_in'] = sto_ins
    merged_ts['bat_in'] = bat_ins
    merged_ts['sto_out'] = sto_outs
    merged_ts['bat_out'] = bat_outs
    merged_ts['sto_soc'] = merged_ts.sto_cap / stoCap
    merged_ts['profit'] = profits
    merged_ts['hoy'] = merged_ts['hoy_y']
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
    pass

#     loopSubFiles(str(argv[1]), int(argv[2]), int(
#         argv[3]), int(argv[4]), int(argv[5]))
