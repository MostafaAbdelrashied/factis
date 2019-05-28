'''
Created on 22.05.2018

@author: mabdelra
'''
import os
import datetime

import csv
import numpy as np
import pandas as pd
from genpy import systools
from genpy.datacrunching import UNIXDataFrameColumn2DataIndex, convert_pd_daterange_to_unixtime, DATAinput2TimeSeries
from Demos.getfilever import fname



def read_spec():
    # Dict that will contain keys and values
    spec_dict = {}
    filename = 'output/spec.txt'
    with open(filename, "r") as f:
        for line in f:
            s = line.strip().split(",")
            spec_dict[s[0]] = s[1]
        return spec_dict


def extract_price_cols(df):
    price_cols = ['Day_Ahead_Auction', 'Intraday_Continuous_Average_Price',
                  'Intraday_Auction_15_minute_call', 'Intraday_Continuous_15_minutes_Average_Price',
                  'Intraday_Continuous_30_minutes_Index_Price']  # Average price for 30 min market is missed up (so instead of this we will use index price
    price_col = []
    for item in price_cols:
        if item in df.columns:
            price_col.append(item)
    return price_col


def resample_timeres(df, time_res='1h'):
    df = UNIXDataFrameColumn2DataIndex(df, unix_column='unixtimestamp')
    df.drop(columns=['YYYYMMDD', 'hhmmss', 'unixtimestamp'], inplace=True)
    for column in df:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    if time_res == '15min':
        last_row = df[-1:]
        last_row.index = last_row.index + pd.DateOffset(hours=1)
        df = df.append(last_row)
        df = df.resample(time_res).ffill()
        df = df[:-1]
    elif time_res == '1h':
        df = df.resample(time_res).mean()

    print('These rows contain NaN values\n{}\nTherefore, they have been eliminated from DF'.format(df[df.isnull().any(axis=1)]))
    df = df.dropna(how='any')

    df.insert(loc=0, column='YYYYMMDD', value=df.index.strftime('%Y%m%d'))
    df.insert(loc=1, column='hhmmss', value=df.index.strftime('%H%M%S'))
    df.insert(loc=2, column='hoy', value=convert_pd_daterange_to_unixtime(df.index))

    return df


def resample_timeres_prices(df, time_res, price_col=[]):
    df = UNIXDataFrameColumn2DataIndex(df, unix_column='unixtimestamp')
    df = df.loc[:, price_col]
    cols = df.columns
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    df = df.resample(time_res).mean()
    return df


def create_ampl_input(input_path, time_res=None):
    df = pd.read_csv(input_path, sep=';\t', engine='python', dtype='str', comment='#', na_values='NaN')
    cols = []
    price_cols = ['YYYYMMDD', 'hhmmss', 'unixtimestamp', 'Day_Ahead_Auction', 'Intraday_Continuous_Average_Price', 'Day_Ahead_Price',
                  'Intraday_Auction_15_minute_call', 'Intraday_Continuous_15_minutes_Average_Price',
                  'Intraday_Continuous_30_minutes_Index_Price']  # Average price for 30 min market is missed up (so instead of this we will use index price
    for item in price_cols:
        if item in df.columns:
            cols.append(item)
            print(item)

    df = df.loc[:, cols]
    if 'Day_Ahead_Price' in df.columns:
        df = df.rename(columns={'Day_Ahead_Price': 'Day_Ahead_Auction'})

    if time_res:
        df = resample_timeres(df, time_res)
        splitted_input = os.path.basename(input_path).split('_')
        output_path = input_path.replace(splitted_input[-1], time_res) + '.dat'
    else:
        splitted_input = os.path.basename(input_path).split('.')
        output_path = input_path.replace(splitted_input[-1], 'dat')

    timestep_number = int(len(df.index))
    param_index = range(1, timestep_number + 1)
    df.insert(loc=0, column='param:', value=param_index)
    #df.rename(columns={'Day_Ahead_Auction': 'day_ahead_auction'}, inplace=True)
    if 'unixtimestamp' in df.columns:
        df.rename(columns={'unixtimestamp': 'hoy'}, inplace=True)

    #print('The head of the Dataframe is\n\n' + str(df.head()) + '\n\n')
    #print('The tail of the Dataframe is\n\n' + str(df.tail()) + '\n\n')

    df[':='] = ''
    #df.drop(columns  = ['YYYYMMDD', 'hhmmss'], inplace = True)
    df.to_csv(output_path, sep=' ', index=False)

    with open(output_path, 'a') as file:
        file.write(';\nparam T := {0};'.format(timestep_number))

    print('Converting\n\n{0}\n\nto\n\n{1}\n\nhas been accomplished successfully'.format(input_path, output_path))
    return df


def dayahead_intraday_combined_AMPL_input(fname1, fname2, price_col1=['Day_Ahead_Auction'], price_col2=['Intraday_Auction_15_minute_call']):
    df_cols = ['YYYYMMDD', 'hhmmss', 'unixtimestamp']
    df1 = create_ampl_input(fname1, colnames=df_cols + price_col1, time_res='15min', sample='upsample')
    df2 = create_ampl_input(fname2, colnames=df_cols + price_col2, time_res='15min', sample='upsample')
    df = df1.merge(df2, how='outer')
    df[':='] = ''
    timestep_number = int(len(df.index))
    split_input = os.path.basename(fname1).split('_')
    output_path = fname1.replace(split_input[0], 'combined')
    output_path = output_path.replace(split_input[0], 'combined')
    output_path = os.path.splitext(output_path)[0] + ".dat"
    df.to_csv(output_path, sep=' ', index=False)
    with open(output_path, 'a') as file:
        file.write(';\nparam T := {0};'.format(timestep_number))
    print('Converting\n\n{0} and {1}\n\nto\n\n{2}\n\nhas been accomplished successfully'.format(fname1, fname2, output_path))


def combine_market_prices(path1, path2, time_res='1h'):
    df_1 = pd.read_csv(path1, usecols=['YYYYMMDD', 'hhmmss', 'unixtimestamp', 'Day_Ahead_Auction', 'Intraday_Continuous_Average_Price'], sep=';\t', engine='python', dtype='str', skiprows=range(0, 17), na_values='NaN')
    df_2 = pd.read_csv(path2, usecols=['YYYYMMDD', 'hhmmss', 'unixtimestamp', 'Intraday_Auction_15_minute_call'], sep=';\t', engine='python', dtype='str', skiprows=range(0, 17), na_values='NaN')
    df = resample_timeres(df_1, time_res, price_col='Day_Ahead_Auction')
    df2 = resample_timeres(df_1, time_res, price_col='Intraday_Continuous_Average_Price')
    df3 = resample_timeres(df_2, time_res, price_col='C')
    df['Intraday_Continuous_Average_Price'] = df2['Intraday_Continuous_Average_Price']
    df['Intraday_Auction_15_minute_call'] = df3['Intraday_Auction_15_minute_call']

    timestep_number = int(len(df.index))
    param_index = range(1, timestep_number + 1)
    df.insert(loc=0, column='param:', value=param_index)
    df[':='] = ''

    if 'unixtimestamp' in df.columns:
        df.rename(columns={'unixtimestamp': 'hoy'}, inplace=True)

    print('The head of the Dataframe is\n\n' + str(df.head()) + '\n\n')
    print('The tail of the Dataframe is\n\n' + str(df.tail()) + '\n\n')

    spitted_input = os.path.basename(path1).split('_')
    output_path = path1.replace(spitted_input[-1], time_res) + '_combined.dat'
    df.to_csv(output_path, sep=' ', index=False)

    with open(output_path, 'a') as file:
        file.write(';\nparam T := {0};'.format(timestep_number))

    print('Converting\n\n{0}\n\nto\n\n{1}\n\nhas been accomplished successfully'.format(path1, output_path))

    return df


if __name__ == "__main__":

    day_ahead = 'input/energycharts_downloads_price_2018_en_1h.csv'
    intraday = 'input/energycharts_downloads_price_2018_en_15min.csv'
    intraday30 = 'input/energycharts_downloads_price_2018_en_30min.csv'
    create_ampl_input(input_path=intraday30, time_res=None)
