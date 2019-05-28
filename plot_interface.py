'''
Updated on 12.02.2019

@author: mabdelra
'''
import os
import datetime
from datetime import timedelta
from pathlib import Path
from sys import exit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sp

from datacrunching import UNIXDataFrameColumn2DataIndex
from plot_framework import IMAGEDIR, DATADIR, soc_price_comparison, \
    rename_files_in_folder, exemplary_day_plot_combined, heatmap_market, linear_reg_price
from plot_framework import cheapflexoverview
from plot_framework import readamplres
from plot_framework import profit_per_cycle_comparison
from plot_framework import profit_limit_vs_profit_fullcycles
from plot_framework import exemplary_price_day_plot
# from plot_framework import exemplary_day_plot
from plot_framework import c_rate_vs_profit_fullcycles
from plot_framework import ampl_carpet_plot
from plot_framework import carpet_plot_eex
from plot_framework import eff_vs_profit_fullcycles
from plot_framework import missing_dates
from plot_framework import prices_comparison
from plot_framework import plot_daily_price_spread
from plot_framework import plot_daily_price_spread_stacked
from plot_framework import plot_rollingmeaneex_yearly
from plot_framework import new_profit
from plot_framework import optimizing_markets
from plot_framework import charge_duplicate_remove
from plot_framework import fix_parameter
from plot_framework import resample_res_output
from plot_framework import bat_in_out_comparison
from plot_framework import bat_in_out_comparison_res
from plot_framework import bat_in_out_comparison_res_2
from plot_framework import soc_in_out_comparison_res
from plot_framework import exemplary_day_plot_crates_many
from plot_framework import exemplary_day_plot_effs_many
from plot_framework import exemplary_day_plot_bat
from plot_framework import prices_boxplot
from plot_framework import cutsubfile
from plot_framework import df_occur
from plot_framework import construct_profit_vs_cost_file
from genpy import ies_style

from genpy.seaborn_plotting import IES_SearbornWrapper
from paths import output_folder, input_folder
from plot_framework import carpet_plot_eex

IES_SearbornWrapper().get_ies_seaborn()
# ies_style.set_layout(style='a4arial')
# iescolors = ies_style.ISEColors
# colorlist = iescolors.get_colorlist()
# FIGHIGHT = 13
# FIGWIDTH = 15
# FIGSIZE = [FIGWIDTH, FIGHIGHT]


if __name__ == '__main__':
    fname = 'energycharts_downloads_price_2018_en_1h.csv'

    # prices_comparison(fname, fname2, price_col=None, time_res='W', steps=True)
    prices_boxplot(fname)
    exit()
    df_occur(path='bat')
    # ampl_carpet_plot(fname2, col = 'sto_soc', folder = path)
    bat_in_out_comparison_res(fname2, folder=path2, day_str='10-02-2017')
    exit()
    profit_limit_vs_profit_fullcycles(path=path2, eff='1', crate='4', combined=True)
    exit()
    exit()
    profit_limit_vs_profit_fullcycles(path=path2, eff='1', crate='0.1', combined=True)
    filename1 = 'Profit_&_Full_Cycles_VS_c-rate_between_2.0_and_40.0_Intraday_Auction_15_minute_call.csv'
    csv_fname1 = 'Profit_&_Full_Cycles_VS_c-rate_between_2.0_and_40.0_Intraday_Auction_15_minute_call.csv'
    x = pd.read_csv(output_folder + '/' + filename1, index_col='profit_limit')
    print(x)
    eq_par = np.polyfit(x.full_cycles, x.index, 10)
    eq_par2 = np.polyfit(x.full_cycles, x.total_profit, 10)
    eq_par3 = np.polyfit(x.full_cycles, x.profit_per_cycle, 10)
    eq = np.poly1d(eq_par)
    eq2 = np.poly1d(eq_par2)
    eq3 = np.poly1d(eq_par3)
    print(eq(4500))

    print(eq2(4500))
    print(eq3(4500))
    exit()
    y = eq(x.index)
    plt.plot(x.index, x.full_cycles)
    plt.plot(x.index, y)
    plt.xticks(x.index)
    plt.show()
    print(y)
    # c_rate_vs_profit_fullcycles(path=output_folder, eff='1')
    # eff_vs_profit_fullcycles(path=ts, crate='4')
    # plot_daily_price_spread(filename1, cols= ['Intraday_Auction_15_minute_call'], density = False)
    # bat_in_out_comparison_res(filename1)
    # profit_limit_vs_profit_fullcycles(path=output_folder, eff='1', crate='4')
    # profit_per_cycle_comparison()
    # rename_files_in_folder('output/ref_files')
    # exemplary_day_plot(filename1, day_str='2017-01-02')
    exit()
    # exemplary_day_plot(filename1, day_str='2017-01-01')
    # Intraday_Auction_15_minute_call
    # Day_Ahead_Auction
    # ampl_carpet_plot(filename1)
    # bat_in_out(filename1, day_str='2017-01-01')
    # exemplary_day_plot_crates_many(path=output_folder, day_str='2017-01-01', eff='1', crate='4')
    # exemplary_day_plot_effs_many(path=output_folder, day_str='2017-01-01', eff='1', crate='4')
    # bat_in_out_comparison_res(filename1, day_str='2017-01-01')
    # soc_in_out_comparison_res(filename1, day_str='2017-01-01')
    # df_occur('lithium-ion_from_1980_to_2018.csv','lead-acid_from_1980_to_2018.csv', 'nickel-cadmium_from_1980_to_2018.csv')
