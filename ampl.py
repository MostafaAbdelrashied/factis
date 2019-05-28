"""
!/usr/bin/python
<<<<<<< HEAD

 mabdelrashied 21.02.2018
=======
 mabdelrashied 21.04.2018
>>>>>>> 14905834603e41da068fad31dbd9875e21af7f96
 [1] Input data file or folder with sub-files. For example: AMPL_2012_Referenzdatensatz.dat
 [2] Loop start. For example: first day of the year --> 0 [Default: 0]
 [3] Loop end. For example: last day of the year --> 0. Day 15 of the year --> 15 [Default: 0]
 [4] Correction. Binary digit to define if the optimization result is merged with
    the input data and the real battery operation is calculated. 0 = No, 1 = Yes. [Default = 0]
 [5] Battery Capacity: Used to write the battery capacity in the results file name. [Default = ""]
 """

import csv
import datetime
import getpass
# + local/relative module import
import inspect
import os
from datetime import timedelta
from os import listdir
from os.path import isfile, join
from subprocess import call

import numpy as np
import pandas as pd

# genpy imports
from genpy.datacrunching import UNIXDataFrameColumn2DataIndex, DATAinput2TimeSeries, csv_to_dict, n_comments
from genpy.systools import sorted_list_of_files, delete_files_in_folder, folder_or_file
from paths import *
from plot_framework import readamplres

# -

myFolder = (os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe()))))


# Functions


def csv_to_ts(file_path: str, unix_column: str = 'hoy') -> object:
    """
    Reads CSV with whitespace delimiter and a unix timestamp to pandas dataframe as timeseries.
    Parameters:
    :param file_path: path of the file to be read
    :param unix_column: identifier for the column where to find the unixtimestamp
    Returns:
    :return: df_ts: dataframe with datetime index
    :rtype: object
    """
    # Reading csv/dat extensions and converting its Index to DatetimeIndex
    df = pd.read_csv(file_path, delim_whitespace=True)
    df_ts = UNIXDataFrameColumn2DataIndex(df, unix_column)
    return df_ts


def read_bat_dat(bat_path: str) -> dict:
    """
    Read the battery data file in AMPL format and provide the information as dictionary
    Parameters:
    :param bat_path: file contains all parameters that was built by main.py e.g. maximum capacity, charge/discharge
    losses
    Returns:
    :return: bat_dict: dictionary contains keys/values of the battery data
    """
    bat_filename = '{}/{}'.format(bat_path, bat_dat_filename)
    df = pd.read_csv(bat_filename, sep=' ''|'';', header=None,
                     skiprows=n_comments(bat_filename, '#'), index_col=1, engine='python')
    bat_dict = df.to_dict()[3]
    return bat_dict


def write_spec(file_name: str, folder: bool, specs_str: str, param_dict: dict = None, price_col: str = None,
               ampl_run: str = 'simple.run') -> dict:
    """
    Writing a txt file contains all battery and optimization specifications in form of a dictionary keys/values with
    "," as delimiter
    Parameters:
    :param file_name: sub-files folder/file name
    :param folder: boolean indicates if there are sub-files in a folder or only a file
    :param specs_str: string includes all technical specifications of a battery e.g. capacity, c-rate, efficiency
    :param param_dict: the targeted dictionary of modification
    :param price_col: the optimized market price column string
    :param ampl_run: the AMPL .run file name
    Returns:
    :return: files_spec: dictionary contains all the updated battery specifications
    """
    # Rewrite battery specifications as a dictionary
    if param_dict is None:
        param_dict = {"Error": "No param_dict available!"}
    spec_list = file_name.split('_')
    if folder:
        files_spec = {'period': spec_list[-2], 'overlap': spec_list[-3], 'length': spec_list[-4],
                      'file_name': file_name, 'specs_str': specs_str, 'file_or_folder': 'folder',
                      'price_col': price_col,
                      'ampl_run': ampl_run}
    else:
        files_spec = {
            'file_name': file_name, 'specs_str': specs_str, 'file_or_folder': 'file'}
    files_spec.update(param_dict)
    # Write a text file contains all specification with ',' as delimiter
    w = csv.writer(open(bat_spec_dir, "w", newline="\n", encoding="utf-8"))
    for key, val in files_spec.items():
        w.writerow([key, val])
    print('\nSpecifications written in output/spec.txt\n')
    return files_spec


def write_period(last_SoC: float, input_file: str, input_file_2: str = None) -> str:
    """
    Writing a txt file contains the last state of charge for the optimization process and the name of the executed
    file.
    Parameters:
    :param last_SoC: the battery state of charge at the end of the previous optimization step
    :param input_file: the name of the file under investigation
    :param input_file_2: multiple files at the same time
    Returns:
    :return: filled_template: string contains all data adjusted for each optimization step.
    """
    with open(period_dat_filename, 'w') as period_file:
        if not input_file_2:
            template = '#This file contains all data adjusted for each' \
                       'optimization step.\nparam sto_soc_start := {SoC};\ndata {input};'
            filled_template = template.format(SoC=last_SoC, input=str(input_file).replace('\\', '/'))
        else:
            template = '#This file contains all data adjusted for each optimization' \
                       'step.\nparam sto_soc_start := {SoC};\ndata {input};\ndata {inputex};'
            filled_template = template.format(
                SoC=last_SoC, input=input_file, inputex=input_file_2)
        period_file.write(filled_template)
    return filled_template


def fix_parameter(df: object, date: list = None, df_cols: list = None, file_name: str = 'fix_01.dat') -> object:
    """
    Creating a correction file that contains all variables need to be fixed in the next optimization
    Parameters:
    :param df: Output dataframe which contains all optimization's results
    :param date: Period of fixation
    :param df_cols: a list contains all variables that need to be fixed
    :param file_name: Correction file name
    Returns:
    :return: out_df: Output dataframe contains all corrected variables in a given period of time
    :rtype: object
    """
    # Extracting data of correction
    if df_cols is None:  # specify all variables and parameters that should be fixed during the next simulation
        df_cols = ['bat_change',
                   'sto_soc', 'sto_cap', 'sto_in', 'sto_out',
                   'sto_charge', 'sto_discharge', 'bat_in', 'bat_out']
    if date:
        temp_df = df.loc[date, df_cols]
    else:
        temp_df = df.loc[:, df_cols]

    # Writing correction data as keys/values in AMPL format ans save it as TXT
    out_df = temp_df.reset_index()
    file = open('correct/{}'.format(file_name), 'w')
    for column in df_cols:
        for time, column_value in out_df[column].items():
            file.write('fix {}[{}] := {};\n'.format(column, time + 1, column_value))
    print('Done writing {file_name} file')
    return out_df


def write_profit(step: int) -> float:
    """
    Extract the optimized profit profit_value from the log file, then,  document it down in the profit log file "AMPL Profit"
    Parameters:
    :param step: the file number
    Returns:
    :return: profit_value: AMPL profit
    """
    # Extract the profit profit_value from AMPL's log file
    global profit_value, name
    with open(log_dat_path_server, 'r') as log_file:
        for line in log_file:
            s = line.split(' ')
            for i, j in enumerate(s):
                if j == "objective":
                    name, profit_value = str(s[i]), float(s[i + 1])
                    print('{0:s} profits = {1:f} Euro\n'.format(name, profit_value))
    # Write the name of the file and its AMPL's profit next to it
    open(profit_log, 'a').write("The file {}_{} Optimized Profit = {:f} Euro\n".format(step, name, profit_value))
    return profit_value


def profit_sum():
    """
    Summing up individual files profits in the log file to obtain the total profit for the while optimization and
    append it to the end of the profit log file
    """
    profit_values = []
    with open(profit_log, 'r') as profit_log_file:
        for line in profit_log_file:
            s = line.split(' ')
            for i, j in enumerate(s):
                if j == "Profit":
                    profit = float(s[i + 2])
                    profit_values.append(profit)
    today_time = datetime.datetime.now().isoformat(timespec='seconds')
    open(profit_log, 'a').write(
        "The total profit from the first to last files = {:f} Euro\n###Created on {}\n\n\n".format(sum(profit_values),
                                                                                                   today_time))
    print('The total profit from the first to last files = {:f} Euro\n'.format(sum(profit_values)))


def last_file_number(opt_path: str) -> int:
    """
    Get first element of last (sorted by name) file (not folder) of the name separated by underscore ("_")
    :param opt_path: the directory that contains optimization files
    :return: file number
    """
    files = [f for f in listdir(opt_path) if isfile(join(opt_path, f))]
    files = sorted(files, key=lambda x: int(x.split("_")[0]))
    filename = files.pop()
    return int(filename.split("_")[0])


def parse_generic_output(specDict: dict, step: int = 0) -> object:
    """
    Reading AMPL output file into a dataframe and save it into csv file
    :param specDict: dictionary contains all battery specs
    :param step: the file number
    :return: df_out: Output data frame contain all results
    """
    column_names = ['Name', 'T', 'Value']
    df = pd.read_csv(out_dat_path_server, delimiter='r[''|'r'\,''|'']', header=None, names=column_names,
                     engine='python')
    df.iloc[:, 1] = df.iloc[:, 1].replace("'", '')  # replace quote symbol from column 1

    df_index = df['Name']
    df_body = df[['T', 'Value']]
    df_body_trans = df_body.T
    to_shift = pd.isnull(df_body_trans.iloc[-1])
    df_body_trans.loc[:, to_shift] = df_body_trans.loc[:, to_shift].shift(1)
    df_body_trans_2 = df_body_trans.T

    df_body_trans_2['Name'] = df_index
    df_body_trans_2.fillna('', inplace=True)

    df_body_trans_2 = df_body_trans_2[:-1]

    table = pd.pivot_table(df_body_trans_2, values='Value', index=['T'], columns=['Name'], aggfunc=np.sum)
    table = table.ix[1:]

    if specDict['file_or_folder'] == 'folder':
        original_input_ts = DATAinput2TimeSeries('{0}/{file_name}/{1}_{file_name}'.format(input_folder, step, **specDict),
                                                 unix_column='hoy', escapechar=None, delim_whitespace=True, verbose=0,
                                                 cutendlines=2, skipinitialspace=True)
        output_file = '{0}/{1}_res_{file_name}'.format(parsed_files_dir, step, **specDict)
    else:
        original_input_ts = DATAinput2TimeSeries('{0}/{file_name}'.format(input_folder, **specDict), unix_column='hoy',
                                                 escapechar=None, delim_whitespace=True, verbose=0,
                                                 cutendlines=2, skipinitialspace=True)
        output_file = '{0}/res_{specs_str}_{file_name}'.format(parsed_files_dir, **specDict)

    columns_list = list(original_input_ts.columns.values)

    for item in columns_list:
        if (item != ':=') and (item != 'param:'):
            table[item] = np.array(original_input_ts[item])

    df_out = pd.DataFrame(table.to_records())
    df_out = df_out.apply(pd.to_numeric, errors='ignore')
    df_out.to_csv(output_file, sep=" ")

    print('The parsed file is in: {0}\n'.format(output_file))
    return df_out


def combineResults(folder: str, period: str = 'D', length: int = 1, start: int = 0, end: int = 0,
                   unix_column: str = 'hoy') -> object:
    """
    Combine all the files located in the folder into one single dataframe.
    Parameters
    :param folder: file_path that contains all parsed output files
    :param period: the dataframe period type type e.g. 'D', 'H' ... etc
    :param length: the total period length
    :param start: number of file to start with
    :param end: number of file to end with
    :param unix_column: unix timestamp string
    Returns
    :return: global_df: Dataframe contains all output results with datetime index
    """
    sorted_file_list = sorted_list_of_files(folder)

    if end != 0:
        sorted_file_list = sorted_file_list[:end]
    if start != 0:
        sorted_file_list = sorted_file_list[start:]

    number_of_files = len(sorted_file_list)
    firstFile = sorted_file_list.pop(0)
    print('\nCombining files\nThe first file in the folder is called: {0}\nThere are {1} files to be combined.'.format(
        firstFile, number_of_files))
    file_path = '{0}/{1}'.format(folder, firstFile)
    global_df = subPartinDF(file_path, period, length, unix_column)

    for fileName in sorted_file_list:
        file_path = '{0}/{1}'.format(folder, fileName)
        next_files_sub_part = subPartinDF(file_path, period, length, unix_column)
        try:
            global_df = global_df.append(
                next_files_sub_part, ignore_index=False)
        except AttributeError:
            print("The KeyError caused an AttributeError!")

    return global_df


def write_rsyncTo_sx401(bat_input_dir: str):
    """
    Write a bash file to run optimization with the required files
    :param bat_input_dir: input files directory
    """
    _user = getpass.getuser()

    rsync_file = 'rsyncTo_sx401.sh'

    template = ('#/bin/bash \n\n' +
                'rsync -v -r -t -u simple.run sx401:/net/home/{}/{}/amplopt \n'.format(_user[0], _user) +
                'rsync -v -r -t -u bat.dat sx401:/net/home/{}/{}/amplopt \n'.format(_user[0], _user) +
                'rsync -v -r -t -u period.dat sx401:/net/home/{}/{}/amplopt \n'.format(_user[0], _user) +
                'rsync -v -r -t -u {} sx401:/net/home/{}/{}/amplopt/{} \n'.format(bat_input_dir, _user[0], _user,
                                                                                  bat_input_dir))

    with open(rsync_file, 'w') as rsync_file:
        rsync_file.write(template)


def loopSubFiles_sx401(opt_filename: str, loopStart: int = 0, loopEnd: int = 0, correct: bool = False,
                       specs_values: str = "", paramDict: dict = None, realDataFile: str = None, price_col: str =
                       'Day_Ahead_Auction', extLoadFile: str = "", beginSOC: float = 0.0, simple_run: str =
                       'simple.run', copy_input=False):
    """
    Loops over the files in a folder or just solves a file
    Parameters:
    :param opt_filename: name of the file/Folder folder to work with
    :param loopStart: first file: 0 is first in folder
    :param loopEnd: if None  last in folder is automatically used
    :param correct: 1 if predicted is validated with real data. 0 if only predicted data is used
    :param specs_values: Defined scenario name to keep track of conditions used
    :param paramDict: dictionary. Parameters used in the simulation
    :param realDataFile: File with real PV and Load. Used to calculate real power flows after optimizing with predicted
    :param price_col: Market's price under investigation
    :param extLoadFile: File written in period.dat to be read load data file.
    :param beginSOC: the state of charge of the battery system at the beginning of the new optimization
    :param simple_run: AMPL's .run file
    :param copy_input: boolean indicates whether input files folder should be copied or not (False accelerates the
    process)
    """

    global lastSOC, last_real_soc, original_real_input_ts

    if correct:
        real_input_path = '{0}/{1}'.format(input_folder, realDataFile)
        print('Reading input file: {}\n'.format(real_input_path))
        original_real_input_ts = DATAinput2TimeSeries(real_input_path, unix_column='hoy', sep=None, escapechar=None,
                                                      delim_whitespace=True, verbose=0, cutendlines=2)

        cols = ['hoy', 'Load_Pel', 'PV1_Pel']
        original_real_input_ts = original_real_input_ts[cols]
        original_real_input_ts['hoy'] = original_real_input_ts['hoy'].astype(int)
        original_real_input_ts = original_real_input_ts.resample('1min', how='mean')

    file_or_folder_path = '{0}/{1}'.format(input_folder, opt_filename)
    if paramDict is not None:
        write_spec(opt_filename, folder_or_file(file_or_folder_path), specs_values, paramDict, price_col=price_col,
                   ampl_run=simple_run)
    else:
        write_spec(opt_filename, folder_or_file(file_or_folder_path), specs_values)

    # System characteristics from the battery
    bat_dict = read_bat_dat(myFolder)
    PV_scale = bat_dict['scale_pv']
    load_scale = bat_dict['scale_load']
    sto_nom_cap = bat_dict['sto_cap_max']

    # Create rsync files
    write_rsyncTo_sx401(file_or_folder_path)

    # If correct then read real data from different file
    if folder_or_file(file_or_folder_path):  # if Folder: several files
        delete_files_in_folder(parsed_files_dir)  # 'output/parsed')
        delete_files_in_folder(merged_files_dir)  # 'output/merged')
        specDict = csv_to_dict(bat_spec_dir)
        period = specDict['period']
        length = specDict['length']

        if loopEnd == 0:
            print('{0}/{1}'.format(input_folder, opt_filename))

            loopEnd = last_file_number(
                '{0}/{1}'.format(input_folder, opt_filename))

        for step in range(int(loopStart), int(loopEnd + 1)):
            if step == loopStart:
                lastSOC = beginSOC
                last_real_soc = beginSOC
                print('the first step ({0}) of the optimisation starts with a SOC of {1}'.format(
                    loopStart, lastSOC))

            else:
                if not correct:
                    if lastSOC > 1:
                        lastSOC = 1
                    if lastSOC < 0:
                        lastSOC = 0

                print('the {0} step of the optimisation starts with a SOC of {1}'.format(step, lastSOC))
            if correct:
                pass

            if extLoadFile == '':
                FILE = '{0}/{1}/{2}_{3}'.format(input_folder, opt_filename, step, opt_filename)
                write_period(lastSOC, FILE)
                if os.name == 'nt':  # For Windows
                    # Copying input.dat files from local computer to sx401
                    if copy_input:
                        call([windows_powershell_path, 'cp -r', input_data_path_local, input_data_path_server])
                    # Copying simple.run, bat.dat, period.dat and model.mod files from local computer to sx401
                    call([windows_powershell_path, 'cp', bat_dat_path, opt_server_path])
                    call([windows_powershell_path, 'cp', per_dat_path, opt_server_path])
                    call([windows_powershell_path, 'cp', model_filename, model_filename_server])
                    call([windows_powershell_path, 'cp', simple_run_path, opt_server_path])
                    call([windows_powershell_path, 'cp', fix_file, fix_file_path_server])
                    print("Done copying to sx401")
                    # Run AMPL on the server
                    username_server = getpass.getuser() + '@sx401'
                    call(
                        [windows_powershell_path, 'ssh', username_server, '"', 'cd', 'amplopt', ';', 'ampl',
                         'simple.run', '"'])
                    parse_generic_output(specDict, step)
                    # write_profit(step, opt_filename)
                    print("Done computing on sx401")
                    # Copying  generic_output.dat file from sx401 to local computer
                    call([windows_powershell_path, 'cp', out_dat_path_server, out_dat_path_local])
                    call([windows_powershell_path, 'cp', log_dat_path_server, log_dat_path_local])
                    print("Done copying from sx401")

                else:  # For Linux
                    pass

            else:
                FILE = '{0}/{1}/{2}_{3}'.format(input_folder, opt_filename, step, opt_filename)
                FILEex = '{0}/{1}/{2}_{3}'.format(input_folder, extLoadFile, step, extLoadFile)
                write_period(lastSOC, FILE, FILEex)
                call(['ampl', 'simple.run'])
                call(['{0}/auto_generic2.sh'.format(output_folder),
                      '{0}/{1}_res_{2}'.format(parsed_files_dir, step, opt_filename),
                      '{0}/{1}'.format(generic_folder, 'generic_out.dat'), FILE, str(PV_scale), str(load_scale),
                      FILEex])

            if correct:
                print(
                    'Merging results and input file for the step {0}'.format(step))

                # "output/parsed/", 1)
                parsed_files = sorted_list_of_files(parsed_files_dir, 1)
                last_parsed_file = '{0}/{1}'.format(parsed_files_dir, parsed_files[-1])
                last_real_cap = last_real_soc * sto_nom_cap
                lastCap = lastSOC * sto_nom_cap

                relevant_output_ts = upsampling(subPartinDF(last_parsed_file, period, length), lastCap,
                                                original_resolution='15min', new_resolution='15min')
                relevant_input_ts = cutsubbyts(original_real_input_ts, relevant_output_ts)
                relevant_input_ts = upsampling(relevant_input_ts, last_real_cap,
                                               original_resolution='60min', new_resolution='15min')

                merged_ts = inoutmerge(relevant_input_ts, relevant_output_ts, PV_scale, load_scale,
                                       price_col)
                merged_ts = surplus4merged(merged_ts)
                merged_ts = charge_duplicate_remove(merged_ts, discharge_col='sto_discharge_x', ref_col='bat_change_x')
                merged_ts = charge_duplicate_remove(merged_ts, discharge_col='sto_discharge_y', ref_col='bat_change_y')
                print(
                    'Calculating real battery operation and energy flows for the step {0}\n'.format(step))

                # TEST TU ADD BATTERY DISCHARGE
                # merged_ts = surplus2bat_basic(merged_ts)
                # print(merged_ts.head(10))
                # print(merged_ts.loc[:, ['bat_change_x', 'bat_change_y']])
                # exit()
                merged_ts = surplus2bat(merged_ts, last_real_cap, bat_dict)

                # merged_ts = batPVLoad2newbilance(merged_ts, bat_dict)
                merged_ts.to_csv("{}/{}_merged.dat".format(merged_files_dir, parsed_files[-1]), sep=" ", index=False)
                mergedFILE = '{0}/{1}_res_{2}_merged.dat'.format(merged_files_dir, step, opt_filename)
                cut_sub = subPartinDF(mergedFILE, period, length, 'hoy_x')
                # ACTUALIZE SOC WHEN IT IS SMALLER THAN REAL SOC
                last_real_soc = float(cut_sub.loc[:, 'sto_soc_x'][-1:])
                lastSOC = float(cut_sub.loc[:, 'sto_soc_y'][-1:])
                # Only update lastSoC in case it is below the last_real_soc. This
                # rule helps to add more "charging allowance" when not much sun
                # is expected.
                if last_real_soc > lastSOC:
                    lastSOC = last_real_soc
                # lastSOC = float(cut_sub.sto_soc_x[-1:])  # This would be the standard form to update the value in model predictive control.
                ##################
                if lastSOC > 1:
                    lastSOC = 1
                if lastSOC < 0:
                    lastSOC = 0

            if step == int(loopEnd):
                print('Successfully looped through all files of {0}. Specifications located in output/spec.txt'.format(
                    opt_filename))

        global_df = combineResults(parsed_files_dir, period, length)
        final_filename = 'res_{specs_str}_{input_dat_1}{input_dat_2}_{price_col}.dat'.format(**specDict)
        df_output_path = '{}/{}'.format(output_folder, final_filename)
        global_df.to_csv(df_output_path, sep=',')
        print('The combined file is in: {0}/{1}\n'.format("output", final_filename))

        if correct:
            global_df = combineResults(merged_files_dir, period, length, 0, 0, 'hoy_x')
            final_filename = 'res_{specs_str}_{input_dat_1}{input_dat_2}{input_real_dat}_merged.dat'.format(
                **specDict)
            global_df.to_csv('{}/{}'.format(output_folder, final_filename), sep=',')
            print(
                'The combined file is in: {0}/{1}\n'.format("output", final_filename))

    if not folder_or_file(file_or_folder_path):  # if File
        lastSOC = beginSOC
        FILE = '{0}/{1}'.format(input_folder, opt_filename)
        write_period(lastSOC, FILE)
        call(['ampl', 'simple.run'])
        newfilename = 'res_{0}_{1}'.format(specs_values, opt_filename)
        call(['{0}/auto_generic.sh'.format(output_folder), '{0}/{1}'.format(output_folder, newfilename),
              '{0}/{1}'.format(generic_folder, 'generic_out.dat'), FILE, str(PV_scale), str(load_scale)])

        if correct:
            print('Merging results and input file')

            last_parsed_file = '{0}/{1}'.format(output_folder, newfilename)
            lastCap = lastSOC * sto_nom_cap
            relevant_output_ts = upsampling(
                csv_to_ts(last_parsed_file, 'hoy'), lastCap)
            relevant_input_ts = cutsubbyts(
                original_real_input_ts, relevant_output_ts)
            merged_ts = inoutmerge(
                relevant_input_ts, relevant_output_ts, PV_scale, load_scale)
            merged_ts = surplus4merged(merged_ts)
            print('Calculating real battery operation and energy flows\n')

            # TEST TU ADD BATTERY DISCHARGE
            #  merged_ts = surplus2bat_basic(merged_ts)
            merged_ts = surplus2bat(merged_ts, lastCap, bat_dict)
            #################
            merged_ts = batPVLoad2newbilance(merged_ts, sto_nom_cap)
            newfilename = newfilename[:-4]
            merged_ts.to_csv(
                "{0}/{1}_merged.dat".format(output_folder, newfilename), sep=",")

        print(
            'Successfully looped through all files of {0}. Specifications located in output/spec.txt'.format(
                opt_filename))


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
    bat_dict = read_bat_dat(myFolder)
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
    df_ts = csv_to_ts(path, unix_column)
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
    # maxnumberoffill = int(original_resolution[:-3]) - int(new_resolution[:-3])
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
    period_ts = pd.read_csv(period_fname, ' ', index_col='hoy', skipfooter=2, engine='python')
    tobecutted_ts = readamplres(tobecutted_fname, unix_column='hoy',
                                drop_zeros=False,
                                drop_str=False)
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
