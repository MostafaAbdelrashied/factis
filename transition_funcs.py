'''
Created on 18.09.2018

@author: mabdelra
'''


import os
import math
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import reduce
from datetime import datetime, timedelta
from create_ampl_input import resample_timeres, resample_timeres_prices, extract_price_cols
from genpy import datacrunching
import pytz
from genpy.datacrunching import DATAinput2TimeSeries, UNIXDataFrameColumn2DataIndex
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from genpy import ies_style
from genpy.plottools import savefig2multiplepaths, carpet
from genpy.plottools import addmanuallegend
from genpy.plottools import Multilingual
from genpy import datacrunching as dtc
from plot_framework import readamplres
from paths import output_folder
from systools import delete_files_in_folder


def fix_ampl_input(fname, fix_cols=['bat_in', 'bat_out'], resample=False):
    df = readamplres(fname, unix_column='hoy', drop_str=False, drop_zeros=False)[fix_cols]
    if resample:
        df = df.resample('15min').ffill()
    df.set_index(np.arange(1, len(df) + 1), inplace=True)
    delete_files_in_folder('correct')
    for column in fix_cols:
        for time, value in df[column].iteritems():
            with open('correct/fix.dat', "a") as fix_file:
                fix_file.write('fix {}[{}] := {};\n'.format(column, time, value))
    print('Successfully Done!')


if __name__ == '__main__':
    fname = 'output/res_0_1000_0_0_1_1_energycharts_downloads_price_2017_en_1h.dat_365.0_0.0_D_subFiles_Day_Ahead_Auction.dat'
    fix_ampl_input(fname, fix_cols=['bat_in', 'bat_out'], resample=True)
