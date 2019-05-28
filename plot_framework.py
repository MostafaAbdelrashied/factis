'''
Created on 02.11.2018

@author: mabdelra
'''
import os
import shutil
from datetime import datetime, timedelta

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import pytz
import seaborn as sns

from create_ampl_input import extract_price_cols
from genpy import datacrunching
from genpy import synpro_tools as syntools
from genpy import ies_style
from genpy.datacrunching import DATAinput2TimeSeries, UNIXDataFrameColumn2DataIndex, read_dat_to_timeseries
from genpy.plottools import Multilingual
from genpy.plottools import addmanuallegend
from genpy.plottools import savefig2multiplepaths, carpet
from paths import plot_folder, output_folder, input_folder

ies_style.set_layout(style='a4arial')
iescolors = ies_style.ISEColors
colorlist = iescolors.get_colorlist()

FIGWIDTH = 6.3 * 2
FIGHIGHT = FIGWIDTH / 3
FIGSIZE = [FIGWIDTH, FIGHIGHT]
FIGSIZE_carpet = [16, 16 / 3]

DATADIR = 'output'
# DATADIR = 'output/new_results/03. Final Final/24.0_12.0_H/Combined'
IMAGEDIR = 'output/plots'


def charge_duplicate_remove(df_temp, discharge_col='sto_discharge', ref_col='bat_change'):
    for index, rows in df_temp.iterrows():
        if df_temp.loc[index, ref_col] == 0:
            df_temp.loc[index, discharge_col] = 0

    return df_temp


def resample_res_output(*fnames, new_res='15min'):
    for fname in fnames:
        df = readamplres("{0}/{1}".format(DATADIR, fname), unix_column='hoy', drop_zeros=False,
                         drop_str=False)
        price_col = fname.split('_subFiles_')[1].replace(".dat", "")
        df = charge_duplicate_remove(df, discharge_col='sto_discharge', ref_col='bat_change')
        df = df.resample(new_res).ffill()
        df.insert(0, 'T', range(1, len(df) + 1))
        df['hoy'] = df.index.astype(np.int64) // 10 ** 9
        f_out = 'res_0_1000_0_0_4_1_energychartss_downloads_price_2017_en_15min.dat_24.0_12.0_H_subFiles_Day_Ahead_Auction.dat'
        df.to_csv("{0}/{1}".format(DATADIR, f_out), sep=',')
        print('Done!\n')


def cutsubfile(tobecutted_fname, period_fname):
    """
    Cut the the relevant part from the output file to be simulated accordingly with the revelant part from the input
    """
    period_ts = pd.read_csv(period_fname, ' ', index_col='hoy', skipfooter=2, engine='python')
    tobecutted_ts = df = readamplres(tobecutted_fname, unix_column='hoy',
                                     drop_zeros=False,
                                     drop_str=False)
    cutted_ts = tobecutted_ts.loc[tobecutted_ts['hoy'].isin(period_ts.index)]
    return cutted_ts


def fix_parameter(df, date=None, df_cols=None, fname='fix_01.dat'):
    if df_cols is None:
        df_cols = ['bat_change',
                   'sto_soc', 'sto_cap', 'sto_in', 'sto_out',
                   'sto_charge', 'sto_discharge', 'bat_in', 'bat_out']
    if date:
        temp_df = df.loc[date, df_cols]
    else:
        temp_df = df.loc[:, df_cols]

    df2 = temp_df.reset_index()
    file = open('correct/{}'.format(fname), 'w')
    for column in df_cols:
        for time, value in df2[column].items():
            file.write('fix {}[{}] := {};\n'.format(column, time + 1, value))
    print('Done writing {fname} file')


def filled_step(ax, xvalues, yvalues, bottoms=None, color='grey', alpha=0.5, **kwargs):
    '''
    Draw a stepped plot with filled area between data and ts-axis

    :param ax: Axes
                The axes to plot to
    :param xvalues: series of DatetimeObjects
                The index of the data to plot
    :param yvalues: series
                The data to plot
    :param bottoms: series or float
                The bottom of the area to fill out
    :param color: color
                Color to plot in. Element of colorlist[..] / '#920310' / 'black' / ...
    :param alpha: float
                Alpha value of the filled area
    '''
    kwargs.setdefault('step', 'post')

    xvalues = pd.Series(xvalues, index=xvalues, name="Time")

    xvalues_df = pd.DataFrame(xvalues)

    widthlist = datacrunching.getwidths(xvalues.index)
    pd_widths = pd.Series(widthlist, index=xvalues, name="Widths")
    xvalues_df["Widths"] = pd_widths

    try:
        # can not handle clock changes in the date range.
        xvalues_df["deltaxvalues"] = xvalues_df["Time"] + xvalues_df["Widths"]
    except:
        # if clock changes in date range, consider time stamps without timezone awareness.
        temptime = [str(item).split("+")[0] for item in xvalues]  # ['2015-01-01 00:00:00' , ...]
        subtracttime = [str(item).split("+")[1] for item in xvalues]  # ['01:00' , ...]
        temptime_datetime = [datetime.strptime(strdate, '%Y-%m-%d %H:%M:%S') for strdate in temptime]
        converted_time = []
        for i in range(len(temptime_datetime)):
            converted_time.append(temptime_datetime[i] - timedelta(hours=int(subtracttime[i][1])))
        xvalues_df["Time"] = converted_time

        xvalues_df["deltaxvalues"] = xvalues_df["Time"] + xvalues_df["Widths"]

    xsteps = np.ravel(zip(xvalues_df.Time, xvalues_df.deltaxvalues))

    if bottoms is None:
        bottoms = np.zeros_like(yvalues)
    if np.isscalar(bottoms):
        bottoms = np.ones_like(yvalues) * bottoms

    ysteps = np.ravel(zip(yvalues, yvalues))
    bottomsteps = np.ravel(zip(bottoms, bottoms))

    return ax.fill_between(xsteps, ysteps + bottomsteps, bottomsteps, where=ysteps + bottomsteps >= bottomsteps,
                           alpha=alpha, facecolor=color, **kwargs)


def carpetCheapflex(data, zlabel, sampling_step_width_sec, bw=False, figsize=[18, 15], savepath=None, show=True,
                    title=None, language='e', tight=True, carpet_scale_limits=[-10, 100]):
    # Select language
    languagedict = Multilingual(['xlabel', 'ylabel'],
                                {'g': [r'Stunde des Tages', r'Tag des Jahres'],
                                 'e': [r'Hour of the Day', r'Day of the Year']},
                                chosenlanguage=language)
    languagedict = languagedict.chosen

    font = {'size': 200}
    matplotlib.rc('font', **font)

    # Look for the number of samples per hour
    samples_per_hour = 3600.0 / sampling_step_width_sec

    # Generate the carpet_data Matrix
    data = data[12 * 4:]
    last_good_index = int(
        (len(data) - (np.mod(len(data), samples_per_hour * 24))))
    data = data[:last_good_index]
    # data[0] = 6.5
    # carpet_data = np.reshape(data, (samples_per_hour * 24, -1), order='F')
    carpet_data = np.values.reshape(data, (-1, samples_per_hour * 24), order='C')

    # Plot and save it
    fig, (ax1) = plt.subplots(
        1, 1, sharex=False, sharey=False, figsize=figsize)
    if not bw:
        if carpet_scale_limits == [0, 0]:
            plt.imshow(carpet_data, aspect='auto', cmap='YlOrBr')

        else:
            plt.imshow(carpet_data, aspect='auto', vmin=carpet_scale_limits[
                0], vmax=carpet_scale_limits[1], cmap='YlOrBr')
    else:
        if carpet_scale_limits == [0, 0]:
            plt.imshow(carpet_data, aspect='auto', cmap='Greys')
        else:
            plt.imshow(carpet_data, aspect='auto', cmap='Greys', vmin=carpet_scale_limits[
                0], vmax=carpet_scale_limits[1])
    #     plt.yticks([w * int(samples_per_hour)
    #                 for w in range(24)], [w for w in range(24)])
    plt.xticks([w * int(samples_per_hour)
                for w in range(24)], [w for w in range(24)])

    if title is not None:
        plt.title(title)
    plt.xlabel(languagedict['xlabel'])
    plt.ylabel(languagedict['ylabel'])
    cb = plt.colorbar()
    cb.set_label(zlabel)
    savefig2multiplepaths(fig, savepath=savepath, show=show, tight=tight)

    return (fig, ax1)


def readamplres(filepath, unix_column='UNIX_TIME', drop_str=False, drop_zeros=False):
    df = pd.read_csv(filepath, sep=',')

    # Drop first 2 columns: 'Unnamed 0' and 'Unnamed 0.1'
    df.drop(df.columns[[0, 1]], axis=1, inplace=True)

    df = UNIXDataFrameColumn2DataIndex(df, unix_column)
    df.sort_index(inplace=True)
    #     if drop_str is True:
    #         df.drop(columns='solve', inplace=True)

    if drop_zeros is True:
        df = df.loc[:, (df != 0).any(axis=0)]

    return df


def readcsvfile(filepath, unix_column='unixtimestamp', drop_str=False, drop_zeros=False):
    df = pd.read_csv(filepath, sep=';\t', engine='python', comment='#')
    df = UNIXDataFrameColumn2DataIndex(df, unix_column)
    df.sort_index(inplace=True)
    # df.drop(df.columns[3:8], axis=1, inplace=True)
    return df


def cheapflexoverview(df, dyntarif='incentive_Eur_MWh', chp_on='chp_motor_onCHP1', boiler_on='boiler_onB1',
                      soc='stosor_soc_th', chp_th='chp_th', boiler_th='boiler_th', sto_th='stosor_out_th',
                      load_th='load_th', figsize=FIGSIZE, savepath=None, show=False):
    '''
    Overview plot for AMPL output specified for the CheapFlex project
    :param df:
    :param dyntarif:
    :param chp_on:
    :param boiler_on:
    :param soc:
    :param chp_th:
    :param boiler_th:
    :param sto_th:
    :param load_th:
    :param figsize:
    :param savepath:
    '''

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=False, figsize=figsize)

    # subplot 1 #######################################################################################################
    addmanuallegend(ax1, [r'CHP', r'Boiler', 'Storage'], colors=[colorlist[0], colorlist[1], colorlist[2]],
                    alphas=[0.8, 0.8, 0.8],
                    edgecolors=['white', 'white', 'white'], ncol=3, loc=1, facealpha=0.95)

    filled_step(ax1, df.index, df[chp_th], bottoms=None,
                color=colorlist[0], alpha=0.7)
    filled_step(ax1, df.index, df[boiler_th], bottoms=df[chp_th],
                color=colorlist[1], alpha=0.7)
    filled_step(ax1, df.index, df[sto_th], bottoms=df[chp_th] + df[boiler_th],
                color=colorlist[2], alpha=0.7)

    ax1.plot(df.index, df[load_th], color='black', alpha=1, lw=0.7, ls='steps-post')

    ax1.set_ylabel(r'Last thermisch in kW')

    # subplot 2 #######################################################################################################
    addmanuallegend(ax2, [r'Speicher SoC'], colors=[colorlist[4]], alphas=[1], edgecolors=['white'], ncol=3, loc=1,
                    facealpha=0.95)
    ax2.plot(df.index, df[soc], color=colorlist[4], alpha=1, lw=1, ls='steps-post')

    ax2.set_ylabel(r'Speicher SoC')

    # subplot 3 #######################################################################################################
    ax4 = ax3.twinx()
    addmanuallegend(ax4, ['CHP ON', 'Boiler ON'], colors=[colorlist[0], colorlist[1]], alphas=[1, 1],
                    edgecolors=['white', 'white'], ncol=2, loc=2, facealpha=0.95)
    addmanuallegend(ax3, [r'Anreizsignal'], colors=[colorlist[3]], alphas=[1],
                    edgecolors=['white'], ncol=1, loc=1, facealpha=0.95)

    filled_step(ax4, df.index, df[chp_on], color=colorlist[0])
    filled_step(ax4, df.index, df[boiler_on], color=colorlist[1])
    plt.setp(ax4.get_yticklabels(), visible=False)

    ax3.plot(df.index, df[dyntarif], color=colorlist[3], alpha=1, lw=1, ls='steps-post')
    ax3.set_ylabel(r'Preissignal in Cent/kWh')

    # general settings ################################################################################################
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), rotation='45')

    savefig2multiplepaths(fig, savepath, show=show)


def cheapFlexCompareSzenarios(df1, df2, df3, df4, dyntarif='incentive_Eur_MWh', chp_on='chp_motor_onCHP1',
                              soc='stosor_soc_th', figsize=FIGSIZE, savepath=None):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, sharey=False, figsize=figsize)

    ######################################################################################################
    # Subplot 1, Constant
    ax5 = ax1.twinx()
    ax9 = ax1.twinx()

    filled_step(ax5, df1.index, df1[chp_on], color=colorlist[0], label=r'CHP ON')
    plt.setp(ax5.get_yticklabels(), visible=False)
    ax5.grid(False)

    ax1.plot(df1.index, df1[dyntarif] / 10., color=colorlist[3], alpha=1,
             lw=1, ls='steps-post', label=r'Incentive: Constant')
    ax1.set_ylabel(r'[Cent/kWh]')
    ax1.set_ylim([0, 10])
    # ax1.set_ylim([3, 3.5])

    ax9.plot(df1.index, df1[soc], color=colorlist[4], alpha=1, lw=2, ls='--', label=r'SOC storage')
    ax9.set_ylabel(r'SOC storage')
    ax9.yaxis.set_ticks(np.arange(0, 1, 0.2))

    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax5.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    lines3, labels3 = ax9.get_legend_handles_labels()
    ax9.legend(lines2 + lines + lines3, labels2 + labels + labels3, loc=2, ncol=3)

    ######################################################################################################
    # Subplot 2, HTNT_static
    ax6 = ax2.twinx()
    ax10 = ax2.twinx()

    filled_step(ax6, df2.index, df2[chp_on], color=colorlist[0], label=r'CHP ON')
    plt.setp(ax6.get_yticklabels(), visible=False)
    ax6.grid(False)

    ax2.plot(df2.index, df2[dyntarif] / 10., color=colorlist[3], alpha=1,
             lw=1, ls='steps-post', label=r'Incentive: HT/LT static')
    ax2.set_ylabel(r'[Cent/kWh]')
    ax2.set_ylim([0, 10])

    ax10.plot(df2.index, df2[soc], color=colorlist[4], alpha=1, lw=2, ls='--', label=r'SOC storage')
    ax10.set_ylabel(r'SOC storage')
    ax10.yaxis.set_ticks(np.arange(0, 1, 0.2))

    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax2.get_legend_handles_labels()
    ax10.legend(lines, labels, loc=2, ncol=3)

    ######################################################################################################
    # Subplot 3, Variable Tariff
    ax7 = ax3.twinx()
    ax11 = ax3.twinx()

    filled_step(ax7, df3.index, df3[chp_on], color=colorlist[0], label=r'CHP ON')
    plt.setp(ax7.get_yticklabels(), visible=False)
    ax7.grid(False)

    ax3.plot(df3.index, df3[dyntarif] / 10., color=colorlist[3],
             alpha=1, lw=1, ls='steps-post', label=r'Incentive: Variable Tariff')
    ax3.set_ylabel(r'[Cent/kWh]')
    ax3.yaxis.set_ticks(np.arange(0, 11, 2))
    #    ax3.yaxis.set_ticks(np.arange(2, 6, 1))

    ax11.plot(df3.index, df3[soc], color=colorlist[4], alpha=1, lw=2, ls='--', label=r'SOC storage')
    ax11.set_ylabel(r'SOC storage')
    ax11.yaxis.set_ticks(np.arange(0, 1, 0.2))

    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax3.get_legend_handles_labels()
    ax11.legend(lines, labels, loc=2, ncol=3)

    ######################################################################################################
    # Subplot 4, HTNT_static
    ax8 = ax4.twinx()
    ax12 = ax4.twinx()

    filled_step(ax8, df4.index, df4[chp_on], color=colorlist[0], label=r'CHP ON')
    plt.setp(ax8.get_yticklabels(), visible=False)
    ax8.grid(False)

    ax4.plot(df4.index, df4[dyntarif] / 10., color=colorlist[3], alpha=1,
             lw=1, ls='steps-post', label=r'Incentive: HT/LT dynamic')
    ax4.set_ylabel(r'[Cent/kWh]')
    ax4.yaxis.set_ticks(np.arange(0, 11, 2))
    # ax4.yaxis.set_ticks(np.arange(1, 6, 1))

    ax12.plot(df4.index, df4[soc], color=colorlist[4], alpha=1, lw=2, ls='--', label=r'SOC storage')
    ax12.set_ylabel(r'SOC storage')
    ax12.yaxis.set_ticks(np.arange(0, 1, 0.2))

    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax4.get_legend_handles_labels()
    ax12.legend(lines, labels, loc=2, ncol=3)

    # general settings ################################################################################################
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=True, rotation='45')

    savefig2multiplepaths(fig, savepath, show=True)


def cheapFlexCompareSzenarios_HTVar(df1, df2, df3, df4, dyntarif='incentive_Eur_MWh', chp_on='chp_motor_onCHP1',
                                    soc='stosor_soc_th', figsize=FIGSIZE, savepath=None):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, sharey=True, figsize=figsize)

    # Subplot 1, Constant
    ax5 = ax1.twinx()
    ax9 = ax1.twinx()

    addmanuallegend(ax9, [r'CHP ON'], colors=[colorlist[0]], alphas=[1],
                    edgecolors=['white'], ncol=1, loc=9, facealpha=0.95)
    addmanuallegend(ax9, [r'HT 12h'], colors=[colorlist[3]], alphas=[1],
                    edgecolors=['white'], ncol=1, loc=2, facealpha=1)
    addmanuallegend(ax9, [r'SOC storage'], colors=[colorlist[4]], alphas=[1],
                    edgecolors=['white'], ncol=1, loc=1, facealpha=1, hatches='||')

    filled_step(ax5, df1.index, df1[chp_on], color=colorlist[0])
    plt.setp(ax5.get_yticklabels(), visible=False)
    ax5.grid(False)

    ax1.plot(df1.index, df1[dyntarif] / 10., color=colorlist[3], alpha=1, lw=1, ls='steps-post')
    # ax1.set_ylabel(r'[Cent/kWh]')
    # ax1.set_ylim([1, 5])

    ax9.plot(df1.index, df1[soc], color=colorlist[4], alpha=1, lw=2, ls='--')
    ax9.set_ylabel(r'SOC storage')
    ax9.yaxis.set_ticks(np.arange(0, 1, 0.2))

    # Subplot 2, EEX
    ax6 = ax2.twinx()
    ax10 = ax2.twinx()

    addmanuallegend(ax10, [r'HT Seasonal'], colors=[colorlist[3]], alphas=[1],
                    edgecolors=['white'], ncol=1, loc=2, facealpha=0.95)

    filled_step(ax6, df2.index, df2[chp_on], color=colorlist[0])
    plt.setp(ax6.get_yticklabels(), visible=False)
    ax6.grid(False)

    ax2.plot(df2.index, df2[dyntarif] / 10., color=colorlist[3], alpha=1, lw=1, ls='steps-post')
    # ax2.set_ylabel(r'[Cent/kWh]')
    # ax2.set_ylim([0, 10])

    ax10.plot(df2.index, df2[soc], color=colorlist[4], alpha=1, lw=2, ls='--')
    ax10.set_ylabel(r'SOC storage')
    ax10.yaxis.set_ticks(np.arange(0, 1, 0.2))

    # Subplot 3, HTNT
    ax7 = ax3.twinx()
    ax11 = ax3.twinx()

    addmanuallegend(ax11, [r'HT Monthly'], colors=[colorlist[3]], alphas=[1],
                    edgecolors=['white'], ncol=1, loc=2, facealpha=0.95)

    filled_step(ax7, df3.index, df3[chp_on], color=colorlist[0])
    plt.setp(ax7.get_yticklabels(), visible=False)
    ax7.grid(False)

    ax3.plot(df3.index, df3[dyntarif] / 10., color=colorlist[3], alpha=1, lw=1, ls='steps-post')
    # ax3.set_ylabel(r'[Cent/kWh]')
    # ax3.yaxis.set_ticks(np.arange(2, 6, 1))

    ax11.plot(df3.index, df3[soc], color=colorlist[4], alpha=1, lw=2, ls='--')
    ax11.set_ylabel(r'SOC storage')
    ax11.yaxis.set_ticks(np.arange(0, 1, 0.2))

    # Subplot 4, HTNT_static
    ax8 = ax4.twinx()
    ax12 = ax4.twinx()

    addmanuallegend(ax12, [r'HT Daily'], colors=[colorlist[3]], alphas=[1],
                    edgecolors=['white'], ncol=1, loc=2, facealpha=0.95)

    filled_step(ax8, df4.index, df4[chp_on], color=colorlist[0])
    plt.setp(ax8.get_yticklabels(), visible=False)
    ax8.grid(False)

    ax4.plot(df4.index, df4[dyntarif] / 10., color=colorlist[3], alpha=1, lw=1, ls='steps-post')

    # ax1.set_ylabel(r'Incentive in Cent/kWh')
    # ax1.yaxis.set_ticks(np.arange(1, 6, 1))

    ax12.plot(df4.index, df4[soc], color=colorlist[4], alpha=1, lw=2, ls='--')
    ax12.set_ylabel(r'SOC storage')
    ax12.yaxis.set_ticks(np.arange(0, 1, 0.2))

    # general settings ################################################################################################
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=True, rotation='45')

    savefig2multiplepaths(fig, savepath, show=True)


def plotCHPRuntimeHBar(restable=None, columnname='chp_motor_onCHP1', height=5, space=2, figsize=[10, 8],
                       savepath=None, bw=False):
    """
    Plot horizontal barplot of the chp runtime for different szenarios.
    To do: Read in the correct times

    :param restable:	results table as given out by resanalysis.getsummedenergy()
    :param columnname:	column to be plotted
    :param figsize:		figsize in format (width,height)
    :param savepath:	savepath
    """

    HEIGHT = height
    BOTTOM = space

    EEX = restable[columnname]['priceEurMWh']
    HTNT = restable[columnname]['HTNT']
    CONST = restable[columnname]['constantPrice']

    if bw:
        colorlist = [(0.3, 0.3, 0.3), (0.5, 0.5, 0.5), (0.7, 0.7, 0.7)]

    plt.close("all")
    fig, (ax1) = plt.subplots(1, 1, sharex=True, sharey=True, figsize=figsize)

    # BOTTOM: SCENARIO HT/LT
    ax1.barh(bottom=BOTTOM, width=HTNT, height=HEIGHT, left=None, color=colorlist[0])
    ax1.text(400, BOTTOM + (HEIGHT / 2.), "HT/LT: CHP runtime = {} h".format(HTNT), fontsize=21, fontname='Ubuntu',
             fontweight='normal', horizontalalignment='left', verticalalignment='center')
    # MID: SCENARIO EEX
    ax1.barh(bottom=2 * BOTTOM + HEIGHT, width=EEX, height=HEIGHT, left=None, color=colorlist[1])
    ax1.text(400, 2 * BOTTOM + (3 * HEIGHT / 2.), "EEX: CHP runtime = {} h".format(EEX), fontsize=21, fontname='Ubuntu',
             fontweight='normal', horizontalalignment='left', verticalalignment='center')
    # TOP: SCENARIO CONSTANT
    ax1.barh(bottom=3 * BOTTOM + 2 * HEIGHT, width=CONST, height=HEIGHT, left=None, color=colorlist[2])
    ax1.text(400, 3 * BOTTOM + (5 * HEIGHT / 2.), "Constant: CHP runtime = {} h".format(CONST), fontsize=21,
             fontname='Ubuntu',
             fontweight='normal', horizontalalignment='left', verticalalignment='center')

    # addmanuallegend(ax1, ['EEX', 'HT-LT', 'Constant'], alphas=[1, 1, 1],  # , colors=[colorlist[0], colorlist[1]]
    #                     edgecolors=['white', 'white', 'white'], ncol=1, loc=1, facealpha=0.95)
    ax1.set_xlabel(r'Hours of the year')

    plt.xlim([0, 8760])
    plt.ylim([0, 3 * HEIGHT + 4 * BOTTOM])
    plt.setp(ax1.get_yticklabels(), visible=False)

    savefig2multiplepaths(fig, savepath, show=True)


def thermaloverview(df, dyntarif='price_day_ahead_euro_MWh', chp_th='chp_th', boiler_th='boiler_th',
                    sto_th='stosor_out_th', load_th='load_th', soc='stosor_soc_th', figsize=FIGSIZE, savepath=None):
    """
    blalbal
    :param df:
    :param dyntarif:
    :param chp_th:
    :param boiler_th:
    :param sto_th:
    :param load_th:
    :param soc:
    :param figsize:
    :param savepath:
    """

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=False, figsize=figsize)

    width = 0.0105

    # ax1 = plt.subplot(gs[0,0])

    addmanuallegend(ax1, [r'BHKW', r'Kessel', 'Speicher'], colors=['#179C7D', '#25BAFF', '#EB6A0A'],
                    alphas=[0.8, 0.8, 0.8],
                    edgecolors=['white', 'white', 'white'], ncol=3, loc=1, facealpha=0.95)
    ax1.bar(df.index, df[chp_th], bottom=0, facecolor='#179C7D', edgecolor="none", linewidth=0, alpha=0.7,
            width=width)
    ax1.bar(df.index, df[boiler_th], bottom=df[chp_th], facecolor='#25BAFF', edgecolor="none", linewidth=0, alpha=0.7,
            width=width)
    ax1.bar(df.index, df[sto_th], bottom=df[boiler_th] + df[chp_th], facecolor='#EB6A0A', edgecolor="none", linewidth=0,
            alpha=0.7, width=width)

    ax1.set_ylabel(r'W$\"a$rmeversorgung in kW')

    addmanuallegend(ax2, [r'Speicher SoC'], colors=['#179C7D'], alphas=[1], edgecolors=['white'], ncol=3, loc=1,
                    facealpha=0.95)

    ax2.step(df.index, df[soc], color='#179C7D', alpha=1, lw=0.7)

    ax2.set_ylabel(r'Speicher SoC')

    # ax3 = plt.subplot(gs[2,0])
    addmanuallegend(ax3, ['Anreizsignal'], colors=['#25BAFF'], alphas=[1],
                    edgecolors=['white'], ncol=1, loc=1, facealpha=0.95)

    ax3.step(df.index, df[dyntarif], color='#25BAFF', alpha=1, lw=0.7)
    ax3.set_ylabel(r'Euro/MWh')

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), rotation='45')

    savefig2multiplepaths(fig, savepath)


def barplotCostsGainsCheapFlex(df, figsize=FIGSIZE):
    df['vnne_revenue'] = 10
    df['profit'] = df['gain_el'] - (df['total_chp_costCHP1'] + df['boiler_cost_fuelB1'] + df['boiler_cost_maintB1'])

    fig, ax1 = plt.subplots(1, figsize=figsize)
    bar_width = 0.25
    bar_1 = [i + 1 for i in range(len(df['chp_cost_fuelCHP1']))]
    bar_2 = [i + 1 + bar_width for i in range(len(df['chp_cost_maintCHP1']))]
    bar_3 = [i + 1 + (2 * bar_width) for i in range(len(df['chp_cost_startCHP1']))]
    tick_pos = [i + (1.5 * bar_width) for i in bar_1]

    ax1.bar(bar_1, df['chp_cost_fuelCHP1'], alpha=0.5, color=[colorlist[0]], width=bar_width, label='Erdgaskosten BHKW')
    ax1.bar(bar_1, df['boiler_cost_fuelB1'], bottom=df['chp_cost_fuelCHP1'], alpha=0.5, color=[colorlist[1]],
            width=bar_width, label='Erdgaskosten Boiler')
    ax1.bar(bar_1, df['chp_cost_maintCHP1'], bottom=[i + j for i, j in zip(df['chp_cost_fuelCHP1'],
                                                                           df['boiler_cost_fuelB1'])], alpha=0.5,
            color=[colorlist[2]],
            width=bar_width, label='Wartungskosten BHKW')
    ax1.bar(bar_1, df['boiler_cost_maintB1'], bottom=[i + j + k for i, j, k in zip(df['chp_cost_fuelCHP1'],
                                                                                   df['boiler_cost_fuelB1'],
                                                                                   df['chp_cost_maintCHP1'])],
            alpha=0.5, color=[colorlist[3]],
            width=bar_width, label='Wartungskosten Boiler')
    ax1.bar(bar_1, df['chp_cost_startCHP1'],
            bottom=[i + j + k + l for i, j, k, l in zip(df['chp_cost_fuelCHP1'], df['boiler_cost_fuelB1'],
                                                        df['chp_cost_maintCHP1'], df['boiler_cost_maintB1'])],
            alpha=0.5, color=[colorlist[4]],
            width=bar_width, label='Startkosten BHKW')
    ax1.bar(bar_2, df['incentive_revenue'], alpha=0.5, color=[colorlist[5]], width=bar_width, label='EEX-Einnahmen')
    ax1.bar(bar_2, df['kwk_revenue'], bottom=df['incentive_revenue'], alpha=0.5, color=[colorlist[6]],
            width=bar_width, label='KWKG-Verguetung')
    ax1.bar(bar_2, df['vnne_revenue'], bottom=[i + j for i, j in zip(df['incentive_revenue'],
                                                                     df['kwk_revenue'])], alpha=0.5,
            color=[colorlist[7]],
            width=bar_width, label='vNNE-Einnahmen')
    ax1.bar(bar_3, df['profit'], alpha=0.5, color=[colorlist[8]], width=bar_width,
            label='Differenz Einnahmen zu Kosten (Profit)')
    ax1.set_xlim(1000, 4000)
    ticks_label = df.index
    plt.legend(loc='best', framealpha=0.5, fancybox=True)
    plt.xticks(tick_pos, ticks_label, rotation=0)
    plt.title(r'Einnahmen-, Kosten- / Profitvergleich der Speichergr$\"o\ss$en')
    plt.xlim([min(tick_pos) - bar_width * 2, max(tick_pos) + bar_width * 2])
    ax1.set_xlabel(r'Speichergr$\"o\ss$e in kWh')
    ax1.set_ylabel('Einnahmen / Kosten / Profit in euro ')
    plt.show()


def carpetPlotsCheapflex():
    filename = 'input/amplopt_price_Eur_MWh_mod.dat'
    savepath = 'output/plots/carpet_eex.png'
    data = pd.read_csv(filename, sep=' ')
    data = data['incentive_Eur_MWh'].values / 10

    carpetCheapflex(
        data, zlabel='EEX Day Ahead Price [Cent/kWh]',
        sampling_step_width_sec=900, carpet_scale_limits=[-1.0, 7.0], bw=False, savepath=savepath)

    filename = 'input/amplopt_HT_NT_mod.dat'
    savepath = 'output/plots/carpet_htlt.png'
    data = pd.read_csv(filename, sep=' ')
    data = data['incentive_Eur_MWh'].values / 10

    carpetCheapflex(
        data, zlabel='Dynamic HT/LT incentive [Cent/kWh]',
        sampling_step_width_sec=900, carpet_scale_limits=[2, 4], bw=False, savepath=savepath)


def plot_daily_price_spread(filename, cols=[], figsize=FIGSIZE, density=False):
    p_df = readamplres("{0}/{1}".format(DATADIR, filename), unix_column='hoy', drop_zeros=False, drop_str=False)
    for item in cols:
        maxima = p_df[item].resample('6H').max()
        minima = p_df[item].resample('6H').min()
        df_diff = maxima - minima
        print(df_diff)
        fig, (ax) = plt.subplots(1, 1, sharex=False, sharey=False, figsize=figsize)
        if item == 'sto_soc':
            ax.hist(p_df[item] * 100, bins='auto', label=item, rwidth=1, color=colorlist[0])
            ax.set_xlabel('{} in %'.format(item))
        else:
            ax.hist(df_diff, bins=100, label=item, rwidth=1, color=colorlist[0], density=False, cumulative=False)
            ax.set_xlabel('Maximum 6-Hourly Spread of {} in Euro/MWh'.format(item.replace('_', ' ').title()))

        ax.set_ylabel('ECDF')
        plt.grid(True)
        savefig2multiplepaths(fig, savepath=[plot_folder + '/max_difference_{}_hist'.format(item)], show=True)


def plot_daily_price_spread_stacked(fname, figsize=FIGSIZE):
    df = pd.read_csv(input_folder + '/' + fname, sep=';\t', engine='python', comment='#')
    df = UNIXDataFrameColumn2DataIndex(df, 'unixtimestamp')
    spread = df['Intraday_Auction_15_minute_call'].resample('D').max() - df['Intraday_Auction_15_minute_call'].resample(
        'D').min()
    spread_av = np.mean(spread)
    print(spread_av)
    exit()
    month_to_season_dct = {
        1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn',
        12: 'Winter'
    }
    price_cols = extract_price_cols(df)
    for col in price_cols:
        resampled_df = pd.DataFrame(df[col].resample('D').max() - df[col].resample('D').min())
        grp_ary = [month_to_season_dct.get(t_stamp.month) for t_stamp in resampled_df.index]
        resampled_df['season'] = grp_ary
        winter_df = resampled_df[resampled_df['season'] == 'Winter'][col]
        spring_df = resampled_df[resampled_df['season'] == 'Spring'][col]
        summer_df = resampled_df[resampled_df['season'] == 'Summer'][col]
        autumn_df = resampled_df[resampled_df['season'] == 'Autumn'][col]
        fig, (ax) = plt.subplots(1, 1, sharex=False, sharey=False, figsize=figsize)
        ax.hist([winter_df, spring_df, summer_df, autumn_df],
                bins=20, histtype='bar', rwidth=0.8, alpha=0.5,
                label=['Winter', 'Spring', 'Summer', 'Autumn'])
        ax.set_xlabel('Maximum daily spread of {} price in Euro/MWh'.format(col.replace('_', ' ').title()))
        ax.set_ylabel('Frequency')

        if col in ('Day_Ahead_Auction', 'Intraday_Continuous_Average_Price'):
            xlim = 100
            ylim = 50
            xtick = 5

        else:
            xlim = 200
            ylim = 50
            xtick = 10

        ax.set_xticks(np.arange(0, 400, xtick))
        ax.set_yticks(np.arange(0, 100, 5))
        ax.set_xlim([0, xlim])
        ax.set_ylim([0, ylim])
        ax.text(0.5, 0.95, s='Winter sum: {} Euro/MWh\n'
                             'Spring sum: {} Euro/MWh\n'
                             'Summer sum: {} Euro/MWh\n'
                             'Autumn sum: {} Euro/MWh'.format(int(winter_df.sum()),
                                                              int(spring_df.sum()),
                                                              int(summer_df.sum()),
                                                              int(autumn_df.sum())),
                backgroundcolor='white', horizontalalignment='center',
                multialignment='left', verticalalignment='top',
                transform=ax.transAxes)

        ax.grid(True)
        ax.legend()
        savefig2multiplepaths(fig, savepath=[
            plot_folder + '/season_difference_{}_hist_{}.png'.format(col, fname.split('_')[3])], show=True)


def align_y_axis(ax1, ax2, minresax1, minresax2):
    """ Sets tick marks of twinx axes to line up with 7 total tick marks

    ax1 and ax2 are matplotlib axes
    Spacing between tick marks will be a factor of minresax1 and minresax2"""

    ax1ylims = ax1.get_ybound()
    ax2ylims = ax2.get_ybound()
    ax1factor = minresax1 * 6
    ax2factor = minresax2 * 6
    ax1.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(ax1ylims[0],
                                                                           ax1ylims[1] + (ax1factor -
                                                                                          (ax1ylims[1] - ax1ylims[
                                                                                              0]) % ax1factor) %
                                                                           ax1factor,
                                                                           7)))
    ax2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(ax2ylims[0],
                                                                           ax2ylims[1] + (ax2factor -
                                                                                          (ax2ylims[1] - ax2ylims[
                                                                                              0]) % ax2factor) %
                                                                           ax2factor,
                                                                           7)))


def new_profit(*args):
    for item in args:
        for file in os.listdir(item):
            df = pd.read_csv(item + '/' + file, sep=',', index_col='Unnamed: 0')
            cols = ['bat_in', 'bat_out', 'sto_out', 'sto_in']
            df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
            price_col = file.split('_subFiles_')[1].replace(".dat", "")
            c_rate = float(file.split('_')[5])
            rt_eff = float(file.split('_')[6])
            time_res = file.split('_')[12].replace('.dat', '')
            if time_res == '1h':
                step = 1
            elif time_res == '30min':
                step = 0.5
            elif time_res == '15min':
                step = 0.25

            df['profit'] = df[price_col] * (df['bat_out'] - df['bat_in']) * step / 1000
            df.to_csv('{}/{}'.format(item, file), sep=',')
            print('Done!\n')
    print('All changes were done successfully')


def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1 - y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny + dy, maxy + dy)


def no_eq_full_cycles_and_profit(filename, df):
    price_col = filename.split('_subFiles_')[1].replace(".dat", "")
    nominal_capacity = float(filename.split('_')[2])
    rt_eff = float(filename.split('_')[6])
    step = df.index.to_series().diff().dt.seconds.div(3600, fill_value=0)[1]
    total_power = abs(df['bat_change']).sum()
    eq_fullcycle = total_power * step * rt_eff ** 0.5 / (2 * nominal_capacity)
    eq_fullcycle = round(eq_fullcycle, 2)
    gain = (df[price_col] * df['bat_change'] * step * rt_eff ** 0.5 / 1000).sum()
    gain = round(gain, 2)
    return eq_fullcycle, gain


def no_eq_full_cycles_and_profit_combined(filename, df, var_col='bat_change', profit_col='profit', combined='2'):
    nominal_capacity = float(filename.split('_')[2])
    rt_eff = float(filename.split('_')[6])
    step = df.index.to_series().diff().dt.seconds.div(3600, fill_value=0)[1]
    total_power = abs(df[var_col]).sum()
    eq_fullcycle = total_power * step * rt_eff ** 0.5 / (2 * nominal_capacity)
    eq_fullcycle = round(eq_fullcycle, 2)
    if combined == '2':
        df['profit_o'] = df['Day_Ahead_Auction_o'] * df['bat_change_o'] * step * rt_eff ** 0.5 / 1000
        df['profit'] = df['Intraday_Auction_15_minute_call'] * df['bat_change'] * step * rt_eff ** 0.5 / 1000
        df['profit_c'] = df['profit_o'] + df['profit']
    elif combined == '3':
        df['profit_o_o'] = df['Day_Ahead_Auction_o_o'] * df['bat_change_o_o'] * step * rt_eff ** 0.5 / 1000
        df['profit_o'] = df['Intraday_Auction_15_minute_call_o'] * (
                df['bat_change_o_o'] + df['bat_change_c_o']) * step * rt_eff ** 0.5 / 1000
        df['profit'] = df['Intraday_Continuous_15_minutes_Average_Price'] * df[
            'bat_change'] * step * rt_eff ** 0.5 / 1000
        df['profit_cc'] = df['profit_o_o'] + df['profit_o'] + df['profit']
    gain = df[profit_col].sum()
    gain = round(gain, 2)
    return eq_fullcycle, gain


def optimizing_markets(*fnames):
    appended_data = []
    df_key = []
    for fname in fnames:
        df_temp = readamplres("{0}/{1}".format(DATADIR, fname), unix_column='hoy', drop_zeros=False, drop_str=False)
        price_col = fname.split('_subFiles_')[1].replace(".dat", "")
        acros_dict = {'Day_Ahead_Auction': 'DAA',
                      'Intraday_Continuous_Average_Price': 'IC_1h',
                      'Intraday_Auction_15_minute_call': 'IA_15min',
                      'Intraday_Continuous_15_minutes_Average_Price': 'IC_15min'}
        acronym = acros_dict[price_col]
        df_key.append(acronym)
        # fixed_cols = ['YYYYMMDD', 'hhmmss', 'hoy']
        # variables_col = ['sto_charge', 'sto_discharge', 'bat_in', 'bat_out', 'bat_change', 'sto_in', 'sto_out', 'sto_cap', 'sto_soc', 'profit']
        # col_total_new = [s + '_' + acronym for s in variables_col]
        # df_temp=df_temp.loc[:,variables_col]
        # df_temp.rename(columns={i:j for i,j in zip(variables_col,col_total_new)}, inplace=True)
        appended_data.append(df_temp)
    df = pd.concat(appended_data, axis=1, join='outer', keys=df_key)
    # df = pd.merge(appended_data, on = 'hoy', how='left').fillna(0)
    # df = reduce(lambda ts,y: pd.merge(ts, y,how='inner', on='hoy'), appended_data)
    print(df.head())


def exemplary_day_plot_crates_many(path=output_folder, day_str='2017-01-01', eff='1', crate='0.5', show=False,
                                   combined='1'):
    for fname in os.listdir(path):
        if fname.startswith('res') and fname.endswith('.dat') and fname.split('_')[6] == eff:
            df = readamplres("{0}/{1}".format(DATADIR, fname), unix_column='hoy', drop_zeros=False, drop_str=True)
            price_col = fname.split('_subFiles_')[1].replace(".dat", "")
            if day_str:
                df = df.loc[
                    day_str, [price_col, 'profit', 'sto_out', 'sto_in', 'sto_soc', 'bat_in', 'bat_out', 'bat_change']]
            else:
                df = df.loc[:, [price_col, 'profit', 'sto_out', 'sto_in', 'sto_soc', 'bat_in', 'bat_out', 'bat_change']]
            print(
                'The analysis has been performed on the day {} with the dataframe\n{}\n and found the following results\n'.format(
                    day_str, df.head()))
            c_value = fname.split('_')[5]
            rt_eff = float(fname.split('_')[6])
            if combined == '1':
                eq_fullcycles, gain = no_eq_full_cycles_and_profit(fname, df)
            elif combined == '2':
                eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(fname, df, var_col='bat_change',
                                                                            profit_col='profit_c')
            elif combined == '3':
                eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(fname, df, var_col='bat_change',
                                                                            profit_col='profit_cc')
            print(
                'c-rate = {}\nThe total profit = {:.3f} Euro\nThe number of equivalent full cycles = {:.2f} Cycle'.format(
                    c_value, gain, eq_fullcycles))
            fig, ax = plt.subplots(1, figsize=FIGSIZE, )

            line1 = ax.step(df.index, df['sto_soc'], where='post', label='State of Charge')
            ax.fill_between(df.index, df['sto_soc'], step='post', label='State of Charge', alpha=0.5)
            ax2 = ax.twinx()
            line2 = ax2.step(df.index, df[price_col], where='post', label=price_col.replace('_', ' ').title(),
                             color=colorlist[1], alpha=0.5)
            lines = line1 + line2
            labs = [l.get_label() for l in lines]

            ax.legend(lines, labs, loc='upper center')
            ax.set_xlabel('Time')
            ax.set_ylabel('State of Charge')
            ax.grid(True)
            ax2.set_ylabel('Price in Euro')

            ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 1)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=pytz.timezone('Europe/Berlin')))
            plt.setp(ax.get_xticklabels(), visible=True, rotation='45')

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, horizontalalignment='center')
            ax.set_yticks(np.arange(df['sto_soc'].min(), df['sto_soc'].max() + 0.1, 0.1))
            ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])  # Percentage Conversion

            ax.text(0.12, 0.9,
                    s='Round trip efficiency = {:,.0%}\nC-Rate = {}\nProfit = {:.2f} Euro\nEquivalent full cycles = {:.2f}'.format(
                        rt_eff, c_value, gain, eq_fullcycles),
                    horizontalalignment='center', multialignment='left', verticalalignment='top',
                    transform=ax.transAxes)
            ax.set_title('Exemplary Day: {}'.format(day_str))
            savefig2multiplepaths(fig, savepath=[
                plot_folder + '/crates' + '/exemplary_day_plot_for_{}_{:.2f}_{}.png'.format(day_str, float(c_value),
                                                                                            price_col)], show=show,
                                  tight=True)


def exemplary_day_plot_effs_many(path=output_folder, day_str='2017-01-01', eff='1', crate='0.5', combined='1',
                                 show=False):
    for fname in os.listdir(path):
        if fname.startswith('res') and fname.endswith('.dat') and fname.split('_')[5] == crate:
            df = readamplres("{0}/{1}".format(DATADIR, fname), unix_column='hoy', drop_zeros=False, drop_str=True)
            price_col = fname.split('_subFiles_')[1].replace(".dat", "")
            if day_str:
                df = df.loc[
                    day_str, [price_col, 'profit', 'sto_out', 'sto_in', 'sto_soc', 'bat_in', 'bat_out', 'bat_change']]
            else:
                df = df.loc[:, [price_col, 'profit', 'sto_out', 'sto_in', 'sto_soc', 'bat_in', 'bat_out', 'bat_change']]
            print(
                'The analysis has been performed on the day {} with the dataframe\n{}\n and found the following results\n'.format(
                    day_str, df.head()))
            c_value = fname.split('_')[5]
            rt_eff = float(fname.split('_')[6])
            if combined == '1':
                eq_fullcycles, gain = no_eq_full_cycles_and_profit(fname, df)
            elif combined == '2':
                eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(fname, df, var_col='bat_change',
                                                                            profit_col='profit_c')
            elif combined == '3':
                eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(fname, df, var_col='bat_change',
                                                                            profit_col='profit_cc')
            print(
                'c-rate = {}\nThe total profit = {:.3f} Euro\nThe number of equivalent full cycles = {:.2f} Cycle'.format(
                    c_value, gain, eq_fullcycles))
            fig, ax = plt.subplots(1, figsize=FIGSIZE, )

            line1 = ax.step(df.index, df['sto_soc'], where='post', label='State of Charge')
            ax.fill_between(df.index, df['sto_soc'], step='post', label='State of Charge', alpha=0.5)
            ax2 = ax.twinx()
            line2 = ax2.step(df.index, df[price_col], where='post', label=price_col.replace('_', ' ').title(),
                             color=colorlist[1], alpha=0.5)
            lines = line1 + line2
            labs = [l.get_label() for l in lines]

            ax.legend(lines, labs, loc='upper center')
            ax.set_xlabel('Time')
            ax.set_ylabel('State of Charge')
            ax.grid(True)
            ax2.set_ylabel('Price in Euro')

            ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 1)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=pytz.timezone('Europe/Berlin')))
            plt.setp(ax.get_xticklabels(), visible=True, rotation='45')

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, horizontalalignment='center')
            ax.set_yticks(np.arange(df['sto_soc'].min(), df['sto_soc'].max() + 0.1, 0.1))
            ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])  # Percentage Conversion

            ax.text(0.12, 0.9,
                    s='Round trip efficiency = {:,.0%}\nC-Rate = {}\nProfit = {:.2f} Euro\nEquivalent full cycles = {:.2f}'.format(
                        rt_eff, c_value, gain, eq_fullcycles),
                    horizontalalignment='center', multialignment='left', verticalalignment='top',
                    transform=ax.transAxes)
            ax.set_title('Exemplary Day: {}'.format(day_str))
            savefig2multiplepaths(fig, savepath=[
                plot_folder + '/effs' + '/exemplary_day_plot_for_{}_{:.2f}_{}.png'.format(day_str, float(rt_eff),
                                                                                          price_col)], show=show,
                                  tight=True)

    for fname in fnames:
        df_t = readamplres("{0}/{1}".format(DATADIR, fname), unix_column='hoy', drop_zeros=False, drop_str=True)
        price_col = fname.split('_subFiles_')[1].replace(".dat", "")
        if day_str:
            df = df_t.loc[
                day_str, [price_col, 'profit', 'sto_out', 'sto_in', 'sto_soc', 'bat_in', 'bat_out', 'bat_change']]
        else:
            df = df_t.loc[:, [price_col, 'profit', 'sto_out', 'sto_in', 'sto_soc', 'bat_in', 'bat_out', 'bat_change']]
        # df.set_index(pd.DatetimeIndex(df.index).time, inplace=True)
        print(
            'The analysis has been performed on the day {} with the dataframe\n{}\n and found the following results\n'.format(
                day_str, df.head()))
        c_value = fname.split('_')[5]
        rt_eff = float(fname.split('_')[6])

        if combined == '1':
            eq_fullcycles, gain = no_eq_full_cycles_and_profit(fname, df)
        elif combined == '2':
            eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(fname, df, var_col='bat_change',
                                                                        profit_col='profit_c')
        elif combined == '3':
            eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(fname, df, var_col='bat_change',
                                                                        profit_col='profit_cc')
            print(
                'c-rate = {}\nThe total profit = {:.3f} Euro\nThe number of equivalent full cycles = {:.2f} Cycle'.format(
                    c_value, gain, eq_fullcycles))
        fig, ax = plt.subplots(1, figsize=FIGSIZE)

        line1 = ax.step(df.index, df['sto_soc'], where='post', label='State of Charge')
        ax.fill_between(df.index, df['sto_soc'], step='post', label='State of Charge', alpha=0.5)
        # ax.fill_between(df.index, df['sto_soc'], step='post', label='State of Charge')
        ax2 = ax.twinx()
        line2 = ax2.step(df.index, df[price_col], where='post', label=price_col.replace('_', ' ').title(),
                         color=colorlist[1], alpha=0.5)
        lines = line1 + line2
        labs = [l.get_label() for l in lines]

        ax.legend(lines, labs, loc='upper center')
        ax.set_xlabel('Time')
        ax.set_ylabel('State of Charge')
        ax.grid(True)
        ax2.set_ylabel('Price in Euro')
        # align_yaxis(ax, 0, ax2, 0)
        # sorted(set(df.index.strftime('%H:00:00')))
        # ts = pd.Series(df.resample('1H').mean().index.strftime('%H:%M:%S').tolist())
        # print(ts)

        # print(type(ts.max))
        #     x_ticks = df.resample('1H').mean()
        #     x_ticks.set_index(pd.DatetimeIndex(x_ticks.index).time, inplace=True)
        # x_ticks = df.iloc[::4, :]
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 1)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=pytz.timezone('Europe/Berlin')))
        plt.setp(ax.get_xticklabels(), visible=True, rotation='45')
        # ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 2)))
        # ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M:%S', tz=pytz.timezone('Europe/Berlin')))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, horizontalalignment='center')
        ax.set_yticks(np.arange(df['sto_soc'].min(), df['sto_soc'].max() + 0.1, 0.1))
        ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])  # Percentage Conversion
        # yaxis_step = (df[price_col].max() - df[price_col].min()) / (len(np.arange(df['sto_soc'].min(), df['sto_soc'].max() + 0.1, 0.1)) - 1)
        # ax2.set_yticks(np.arange(0, 60, 10))
        align_yaxis(ax, 0, ax2, 0)
        # ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 1)))
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=pytz.timezone('Europe/Berlin')))

        # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
        # ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        # ax2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace())))

        ax.text(0.15, 0.9,
                s='Round trip efficiency = {:,.0%}\nC-Rate = {}\nProfit = {:.2f} Euro\nEquivalent full cycles = {:.2f}'.format(
                    rt_eff, c_value, gain, eq_fullcycles),
                horizontalalignment='center', multialignment='left', verticalalignment='top', transform=ax.transAxes)
        ax.set_title('Exemplary Day: {}'.format(day_str))
        savefig2multiplepaths(fig, savepath=[
            plot_folder + '/exemplary_day_plot_for_{}_{:.2f}_{}.png'.format(day_str, float(c_value), price_col)],
                              show=True,
                              tight=True)


def exemplary_day_plot_combined(filename, folder=DATADIR, day_str='2017-01-01', combined='1', show=True,
                                figsize=FIGSIZE, tight=True):
    df = readamplres("{0}/{1}".format(folder, filename), unix_column='hoy', drop_zeros=False, drop_str=True)
    sto_soc_temp = ['sto_soc_DA', 'sto_soc_IA', 'sto_soc_IC']
    price_col = ['Day_Ahead_Auction', 'Intraday_Auction_15_minute_call', 'Intraday_Continuous_Average_Price']
    soc_col_out = ['sto_out', 'sto_out_DA', 'sto_out_IA', 'sto_out_IC']
    soc_col_in = ['sto_in', 'sto_in_DA', 'sto_in_IA', 'sto_in_IC']
    df = df.loc[day_str, price_col + soc_col_out + soc_col_in + sto_soc_temp + ['profit', 'sto_soc']]
    df.set_index(pd.DatetimeIndex(df.index).time, inplace=True)
    print(
        'The analysis has been performed on the day {} with the dataframe\n{}\n and found the following results\n'.format(
            day_str, df.head()))
    c_value = filename.split('_')[5]

    if combined == '1':
        eq_fullcycles, gain = no_eq_full_cycles_and_profit(filename, df)
    elif combined == '2':
        eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(filename, df, var_col='bat_change',
                                                                    profit_col='profit_c')
    elif combined == '3':
        eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(filename, df, var_col='bat_change',
                                                                    profit_col='profit_cc')
    print(
        'c-rate = {}\nThe total profit = {:.3f} Euro\nThe number of equivalent full cycles = {:f} Cycle'.format(c_value,
                                                                                                                gain,
                                                                                                                eq_fullcycles))
    fig, ax = plt.subplots(1, figsize=figsize)
    #     sto_soc_DA = (abs(df['sto_out_DA'] -  df['sto_in_DA']) / 1000) * 100
    #     sto_soc_IA = (abs(df['sto_out_IA'] -  df['sto_in_IA']) / 1000) * 100
    #     sto_soc_IC = (abs(df['sto_out_IC'] -  df['sto_in_IC']) / 1000) * 100
    #     sto_soc_temp = pd.DataFrame({'sto_soc_DA': sto_soc_DA, 'sto_soc_IA':sto_soc_IA, 'sto_soc_IC':sto_soc_IC})
    lines = []
    for column in sto_soc_temp:
        line1 = ax.step(df.index, df[column], where='post', label=df[column].name, alpha=0.5)
        lines += line1

    ax2 = ax.twinx()
    for item in price_col:
        line2 = ax2.plot(df.index, df[item], label=item, alpha=0.2)
        lines += line2
    # lines +=  ax.step(df.index, df.sto_soc * 100 , where='post', label = df.sto_soc.name)
    labs = [l.get_label() for l in lines]
    ax.legend(lines, labs, loc=9)

    ax.set_xlabel('Time')
    ax.set_ylabel('State of Charge in %')
    ax.grid(True)
    ax2.set_ylabel('Day Ahead Price in Euro')
    ax.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(15))
    ax2.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(15))
    ax.text(0.12, 0.9, s='c-rate = {}\nProfit = {:.2f} Euro\nFull Cycles = {:f}'.format(c_value, gain, eq_fullcycles),
            horizontalalignment='center', multialignment='left', verticalalignment='top', transform=ax.transAxes)
    ax.set_title('The sample day is {}'.format(day_str))

    savefig2multiplepaths(fig,
                          savepath=[
                              plot_folder + '/exemplary_day_plot_for_{}_{:.2f}.png'.format(day_str, float(c_value))],
                          show=show, tight=tight)


def c_rate_vs_profit_fullcycles(path=output_folder, eff='1', combined='1', show=True, figsize=FIGSIZE, tight=True):
    c_rate_values = [0]
    total_gain = [0]
    total_eq_fullcycles = [0]
    for filename in os.listdir(path):
        if filename.startswith('res') and filename.endswith('.dat') and filename.split('_')[6] == eff:
            price_col = filename.split('_subFiles_')[1].replace(".dat", "")
            df = DATAinput2TimeSeries(os.path.join(path, filename), delim_whitespace=False, unix_column='hoy')
            c_value = filename.split('_')[5]
            c_rate_values.append(float(c_value))
            if combined == '1':
                eq_fullcycles, gain = no_eq_full_cycles_and_profit(filename, df)
            elif combined == '2':
                eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(filename, df, var_col='bat_change',
                                                                            profit_col='profit_c')
            elif combined == '3':
                eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(filename, df, var_col='bat_change',
                                                                            profit_col='profit_cc')
            else:
                print('Please specifiy a correct argument for combined paramter')
            total_gain.append(float(gain))
            total_eq_fullcycles.append(float(eq_fullcycles))

    # total_eq_fullcycles, gain = ['%.2f' % elem for elem in total_eq_fullcycles]
    df = pd.DataFrame(np.column_stack([c_rate_values, total_gain, total_eq_fullcycles]),
                      columns=['c-rate', 'total_profit', 'full_cycles'])
    df.apply(pd.to_numeric, errors='ignore')
    df.set_index('c-rate', inplace=True, drop=False)
    df.sort_index(inplace=True)
    df['profit_per_cycle'] = df['total_profit'] / df['full_cycles']
    df['change_in_profit_per_cycle'] = df['total_profit'].diff() / df['full_cycles'].diff()
    df['LCC'] = 540 * 1000 * df['c-rate']
    df['CC'] = df['LCC'] / 4500
    df['benefit_to_cost_total'] = df['total_profit'] * 100 / df['LCC']
    df['benefit_to_cost_per_cycle'] = df['profit_per_cycle'] * 100 / df['CC']
    df = df.round(decimals=2)
    df.fillna(value=0, inplace=True)
    print('The Dataframe for the total gain and number of full cycles with the change of c-rate is\n\n{}'.format(df))

    # pd.to_numeric([df.index, df['total_profit'], df['full_cycles']], errors='raise')
    fig, ax = plt.subplots(1, figsize=figsize)
    ax2 = ax.twinx()
    # ax3 = ax.twinx()
    ax4 = ax.twinx()

    line1 = ax.plot(df.index, df['total_profit'], linestyle=':', marker='o', label='Profit')
    line2 = ax2.plot(df.index, df['full_cycles'], linestyle=':', marker='o', label='Equivalent Full Cycles',
                     color=colorlist[1])
    # line3 = ax3.plot(df.index, df['profit_per_cycle'], linestyle=':', marker='o', label='Profit per Cycle',
    #                color=colorlist[2])
    # line4 = ax3.plot(df.index, df['change_in_profit_per_cycle'], linestyle=':', marker='o', label='Δ Profit per Δ Fullcycles', color=colorlist[4])
    line5 = ax4.plot(df.index, df['benefit_to_cost_total'], linestyle=':', marker='o', label='Benefit to Cost (%)',
                     color=colorlist[3])

    lines = line1 + line2 + line5  # + line3
    labs = [l.get_label() for l in lines]
    ax.legend(lines, labs, loc='lower right')
    ax.grid()
    ax.set_xlabel('C-Rate')
    ax.set_ylabel('Profit in Euro')
    ax2.set_ylabel('Number of Cycles')
    # ax3.set_ylabel('Euro per Cycle')
    ax4.set_ylabel('BtC Ratio')

    # ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda ts, p: format(int(ts), ',')))
    mkfunc = lambda x, pos: '%1.0fM' % (x * 1e-6) if x >= 1e6 else '%1.0fK' % (x * 1e-3) if x >= 1e3 else '%1.0f' % x
    mkfunc_H = lambda x, pos: '%1.0fH' % (x * 0.01) if x >= 1e1 else '%1.0f' % x
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(mkfunc))
    ax2.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(mkfunc))
    ax.set_xticks(np.arange(0, 4, 0.2))
    ax4.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    ax.set_yticks(np.arange(0, 180000, 20000))
    ax.set_ylim([None, 160000])
    ax2.set_yticks(np.arange(0, 180000, 2000))
    ax2.set_ylim([None, 16000])
    # ax3.set_yticks(np.arange(0, 50, 2))
    # ax3.set_ylim([None, 18])
    ax4.set_yticks(np.arange(0, 100, 2))
    ax4.set_ylim([None, 16])
    align_yaxis(ax, 0, ax2, 0)
    # align_yaxis(ax, 0, ax3, 0)
    align_yaxis(ax, 0, ax4, 0)
    # ax3.spines['right'].set_position(('outward', 80))
    ax4.spines['right'].set_position(('outward', 70))
    savefig2multiplepaths(fig, savepath=[
        plot_folder + '/Profit_&_Full_Cycles_VS_c-rate_between_{}_and_{}_{}.png'.format(min(c_rate_values),
                                                                                        max(c_rate_values), price_col)],
                          show=show, tight=tight)


def construct_profit_vs_cost_file(path=output_folder, show=True, figsize=FIGSIZE, tight=True):
    profit_limits = []
    total_gains = []
    price_cols = []
    total_eq_fullcycles = []
    crates = []
    effs = []
    for filename in os.listdir(path):
        if filename.startswith('res') and filename.endswith('.dat'):
            df = DATAinput2TimeSeries(path + '/' + filename, delim_whitespace=False, unix_column='hoy')
            crate, eff, profit_limit = filename.split('_')[5], filename.split('_')[6], filename.split('_')[7]
            price_col = filename.split('_subFiles_')[1].replace(".dat", "")
            eq_fullcycle, gain = no_eq_full_cycles_and_profit(filename, df)
            profit_limits.append(float(profit_limit))
            total_gains.append(float(gain))
            price_cols.append(price_col)
            total_eq_fullcycles.append(float(eq_fullcycle))
            crates.append(float(crate))
            effs.append(float(eff))

    # total_eq_fullcycles, gain = ['%.2f' % elem for elem in total_eq_fullcycles]
    df = pd.DataFrame(np.column_stack([profit_limits, total_gains, price_cols, total_eq_fullcycles, crates, effs]),
                      columns=['profit_limit', 'total_profit', 'price_col', 'full_cycle', 'crate', 'eff'])
    df.apply(pd.to_numeric, errors='ignore')
    cols = [i for i in df.columns if i not in ["price_col"]]
    for col in cols:
        df[col] = pd.to_numeric(df[col])
    df = df.round(decimals=2)

    df['profit_per_cycle'] = round((df['total_profit'] / df['full_cycle']), 2)
    df['LCC'] = round(540 * 1000 * df['crate'], 2)
    df['CC'] = round(df['LCC'] / 4500, 2)
    df['benefit_to_cost_total'] = round(df['total_profit'] / df['LCC'], 2) * 100
    df['benefit_to_cost_per_cycle'] = round(df['profit_per_cycle'] / df['CC'], 2) * 100
    # df['chnage_in_profit_per_cycle'] = round((df['total_profit'].diff() / df['full_cycles'].diff()), 2)
    lim1 = df['eff'] == 1.0
    lim2 = df['crate'] == 4
    crate_df = df.loc[
        lim1, ['crate', 'full_cycle', 'total_profit', 'benefit_to_cost_total', 'benefit_to_cost_per_cycle']].set_index(
        'crate')
    crate_df.sort_index(inplace=True)

    eff_df = df.loc[
        lim2, ['eff', 'full_cycle', 'total_profit', 'benefit_to_cost_total', 'benefit_to_cost_per_cycle']].set_index(
        'eff')
    eff_df.sort_index(inplace=True)
    print(crate_df)
    print(eff_df)
    fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    ax1.plot(crate_df.index, crate_df.benefit_to_cost_total, 'go--',
             eff_df.index, eff_df.benefit_to_cost_total, 'go-.')
    # ax2.plot(crate_df.index, crate_df.benefit_to_cost_per_cycle, 'bo--',
    #          eff_df.index, eff_df.benefit_to_cost_per_cycle, 'bo-.')
    ax1.set_xlabel('X data')
    ax1.set_ylabel('Total', color='g')
    # ax2.set_ylabel('per Cycle', color='b')

    plt.show()


def profit_per_cycle_comparison(path=output_folder, show=True, figsize=FIGSIZE, tight=True):
    df_lists = []
    vals = ['Intraday', 'Day-Ahead', 'Combined']
    for file in os.listdir(path):
        if file.startswith('Profit') and file.endswith('.csv'):
            df_temp = pd.read_csv(path + '/' + file, index_col='profit_limit')
            df_temp.drop(columns=['profit_limit.1'], axis=1, inplace=True)
            df_lists.append(df_temp)
    df = pd.concat(df_lists, keys=vals)
    fig, ax = plt.subplots(1, figsize=figsize)

    ax.plot(df.loc['Day-Ahead'].index, df.loc['Day-Ahead']['profit_per_cycle'], linestyle=':', marker='o',
            label='Day-Ahead Market')
    ax.plot(df.loc['Intraday'].index, df.loc['Intraday']['profit_per_cycle'], linestyle=':', marker='o',
            label='Intraday Market')
    ax.plot(df.loc['Combined'].index, df.loc['Combined']['profit_per_cycle'], linestyle=':', marker='o',
            label='Combined')

    ax.legend()
    ax.grid()
    ax.set_xlabel('Cost Cycle Limit in Euro per Cycle')
    ax.set_ylabel('Profit per Cycle in Euro')
    ax.set_xticks(df.loc['Combined'].index)
    ax.set_yticks(np.arange(0, 100, 6))
    ax.set_ylim([0, 60])
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    savefig2multiplepaths(fig, savepath=[
        plot_folder + '/Comparison_between_different_markets_{}_and_{}_{}.png'.format(vals[0], vals[1], vals[2])],
                          show=show, tight=tight)


def profit_limit_vs_profit_fullcycles(path=output_folder, eff='1', crate='1', combined='1', show=True, figsize=FIGSIZE,
                                      tight=True):
    profit_limits = []
    total_gain = []
    total_eq_fullcycles = []
    for filename in os.listdir(path):
        if filename.startswith('res') and filename.endswith('.dat') and filename.split('_')[6] == eff and \
                filename.split('_')[5] == crate:
            price_col = filename.split('_subFiles_')[1].replace(".dat", "")
            df = DATAinput2TimeSeries(os.path.join(path, filename), delim_whitespace=False, unix_column='hoy')
            profit_limit = filename.split('_')[7]
            profit_limits.append(float(profit_limit))
            if combined == '1':
                eq_fullcycles, gain = no_eq_full_cycles_and_profit(filename, df)
            elif combined == '2':
                eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(filename, df, var_col='bat_change',
                                                                            profit_col='profit_c')
            elif combined == '3':
                eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(filename, df, var_col='bat_change',
                                                                            profit_col='profit_cc')
            else:
                print('Please specifiy a correct argument for combined paramter')
            total_gain.append(float(gain))
            total_eq_fullcycles.append(float(eq_fullcycles))

    # total_eq_fullcycles, gain = ['%.2f' % elem for elem in total_eq_fullcycles]
    df = pd.DataFrame(np.column_stack([profit_limits, total_gain, total_eq_fullcycles]),
                      columns=['profit_limit', 'total_profit', 'full_cycles'])
    df.apply(pd.to_numeric, errors='ignore')
    df = df.round(decimals=2)
    df.set_index('profit_limit', drop=False, inplace=True)
    df.sort_index(inplace=True)
    df['profit_per_cycle'] = df['total_profit'] / df['full_cycles']
    df['change_in_profit_per_cycle'] = df['total_profit'].diff() / df['full_cycles'].diff()
    df['LCC'] = 540 * 1000 * float(crate)
    df['CC'] = df['LCC'] / 4500
    df['benefit_to_cost_total'] = df['total_profit'] * 100 / df['LCC']
    df['benefit_to_cost_per_cycle'] = df['profit_per_cycle'] * 100 / df['CC']
    df = df.round(decimals=2)
    df.fillna(value=0, inplace=True)
    # slope, intercept = np.polyfit(df.index, df.profit_per_cycle, 1)
    print('The Dataframe for the total gain and number of full cycles with the change of c-rate is\n\n{}'.format(
        df.head()))
    df.to_csv(output_folder + '/Profit_&_Full_Cycles_VS_c-rate_between_{}_and_{}_{}.csv'.format(min(profit_limits),
                                                                                                max(profit_limits),
                                                                                                price_col))
    # print('Slope = {}\nIntercept = {}\n'.format(slope, intercept))

    # pd.to_numeric([df.index, df['total_profit'], df['full_cycles']], errors='raise')
    fig, ax = plt.subplots(1, figsize=figsize)

    ax2 = ax.twinx()
    # ax3 = ax.twinx()
    ax4 = ax.twinx()

    line1 = ax.plot(df.index, df['total_profit'], linestyle=':', marker='o', label='Profit')
    line2 = ax2.plot(df.index, df['full_cycles'], linestyle=':', marker='o', label='Equivalent Full Cycles',
                     color=colorlist[1])
    # line3 = ax3.plot(df.index, df['profit_per_cycle'], linestyle=':', marker='o', label='Profit per Cycle',
    #                  color=colorlist[2])
    # line4 = ax3.plot(df.index, df['change_in_profit_per_cycle'], linestyle=':', marker='o', label='Δ Profit per Δ Fullcycles', color=colorlist[4])
    line5 = ax4.plot(df.index, df['benefit_to_cost_total'], linestyle=':', marker='o', label='Benefit to Cost (%)',
                     color=colorlist[3])

    lines = line1 + line2 + line5  # + line4
    labs = [l.get_label() for l in lines]
    ax.legend(lines, labs, loc='upper right')
    ax.grid()
    ax.set_xlabel('Cost Limit in Euro/Cycle')
    ax.set_ylabel('Profit in Euro')
    ax2.set_ylabel('Number of Cycles')
    # ax3.set_ylabel('Euro per Cycle')
    ax4.set_ylabel('BtC Ratio')
    mkfunc = lambda x, pos: '%1.0fM' % (x * 1e-6) if x >= 1e6 else '%1.0fK' % (x * 1e-3) if x >= 1e3 else '%1.0f' % x
    mkfunc_H = lambda x, pos: '%1.0fH' % (x * 0.01) if x >= 1e1 else '%1.0f' % x
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(mkfunc))
    ax2.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(mkfunc))
    ax4.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.set_xticks(np.arange(0, 42, 2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    #     ax.set_yticks(np.arange(0, 180000, 2000))
    #     ax.set_ylim([None, 18000])
    #     ax2.set_yticks(np.arange(0, 18000, 200))
    #     ax2.set_ylim([None, 1800])
    #     ax3.set_yticks(np.arange(0, 50, 4))
    #     ax3.set_ylim([None, 50])
    ax.set_yticks(np.arange(0, 180000, 20000))
    ax.set_ylim([0, 120000])
    ax2.set_yticks(np.arange(0, 180000, 2000))
    ax2.set_ylim([0, 12000])
    # ax3.set_yticks(np.arange(0, 100, 4))
    # ax3.set_ylim([10, 40])
    ax4.set_yticks(np.arange(0, 100, 1))
    ax4.set_ylim([0, 6])
    align_yaxis(ax, 0, ax2, 0)
    # align_yaxis(ax, 5000, ax3, 4)
    align_yaxis(ax, 0, ax4, 0)

    # ax3.spines['right'].set_position(('outward', 60))
    ax4.spines['right'].set_position(('outward', 70))
    savefig2multiplepaths(fig, savepath=(
        plot_folder + '/Profit_&_Full_Cycles_VS_c-rate_between_{}_and_{}_{}.png'.format(min(profit_limits),
                                                                                        max(profit_limits),
                                                                                        price_col),),
                          show=show, tight=tight)


def eff_vs_profit_fullcycles(path=output_folder, crate='0.5', combined='1', show=True, figsize=FIGSIZE, tight=True):
    eff = [0]
    total_gain = [0]
    total_eq_fullcycles = [0]
    for filename in os.listdir(path):
        if filename.startswith('res') and filename.endswith('.dat') and filename.split('_')[5] == crate:
            price_col = filename.split('_subFiles_')[1].replace(".dat", "")
            df = DATAinput2TimeSeries(os.path.join(path, filename), delim_whitespace=False, unix_column='hoy')

            eff_value = filename.split('_')[6]
            eff.append(float(eff_value) * 100)
            if combined == '1':
                eq_fullcycles, gain = no_eq_full_cycles_and_profit(filename, df)
            elif combined == '2':
                eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(filename, df, var_col='bat_change',
                                                                            profit_col='profit_c')
            elif combined == '3':
                eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(filename, df, var_col='bat_change',
                                                                            profit_col='profit_cc')
            else:
                print('Please specifiy a correct argument for combined paramter')
            total_gain.append(float(gain))

            total_eq_fullcycles.append(float(eq_fullcycles))

    df = pd.DataFrame(np.column_stack([eff, total_gain, total_eq_fullcycles]),
                      columns=['round_trip_efficiency', 'total_profit', 'full_cycles'])
    df.set_index('round_trip_efficiency', drop=False, inplace=True)
    df.sort_index(inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    df['profit_per_cycle'] = df['total_profit'] / df['full_cycles']
    df['change_in_profit_per_cycle'] = df['total_profit'].diff() / df['full_cycles'].diff()
    df['LCC'] = 540 * 1000 * float(crate)
    df['CC'] = df['LCC'] / 4500
    df['benefit_to_cost_total'] = df['total_profit'] * 100 / df['LCC']
    df['benefit_to_cost_per_cycle'] = df['profit_per_cycle'] * 100 / df['CC']
    df = df.round(decimals=2)
    df.fillna(value=0, inplace=True)
    print(
        'The Dataframe for the total gain and number of full cycles with the change of Round Trip Efficiency is\n\n{}'.format(
            df))

    fig, ax = plt.subplots(1, figsize=figsize)

    ax2 = ax.twinx()
    # ax3 = ax.twinx()
    ax4 = ax.twinx()

    line1 = ax.plot(df.index, df['total_profit'], linestyle=':', marker='o', label='Profit')
    line2 = ax2.plot(df.index, df['full_cycles'], linestyle=':', marker='o', label='Equivalent Full Cycles',
                     color=colorlist[1])
    # line3 = ax3.plot(df.index, df['profit_per_cycle'], linestyle=':', marker='o', label='Profit per Cycle',
    #                 color=colorlist[2])
    # line4 = ax3.plot(df.index, df['change_in_profit_per_cycle'], linestyle=':', marker='o', label='Δ Profit per Δ Fullcycles', color=colorlist[4])
    line5 = ax4.plot(df.index, df['benefit_to_cost_total'], linestyle=':', marker='o', label='Benefit to Cost (%)',
                     color=colorlist[3])

    lines = line1 + line2 + line5  # + line3
    labs = [l.get_label() for l in lines]
    ax.legend(lines, labs, loc='upper center')
    ax.grid()
    ax.set_xlabel('Round Trip Efficiency')
    ax.set_ylabel('Profit in Euro')
    ax2.set_ylabel('Number of Cycles')
    # ax3.set_ylabel('Euro per Cycle')
    ax4.set_ylabel('BtC Ratio')

    ax.set_xticks(df.index)
    mkfunc = lambda x, pos: '%1.0fM' % (x * 1e-6) if x >= 1e6 else '%1.0fK' % (x * 1e-3) if x >= 1e3 else '%1.0f' % x
    mkfunc_H = lambda x, pos: '%1.0fH' % (x * 0.01) if x >= 1e1 else '%1.0f' % x
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(mkfunc))
    ax2.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(mkfunc))
    ax4.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    ax.set_yticks(np.arange(0, 180000, 20000))
    ax.set_ylim([None, 160000])
    ax2.set_yticks(np.arange(0, 180000, 2000))
    ax2.set_ylim([None, 16000])
    # ax3.set_yticks(np.arange(0, 50, 2))
    # ax3.set_ylim([None, 18])
    ax4.set_yticks(np.arange(0, 100, 2))
    ax4.set_ylim([None, 16])

    align_yaxis(ax, 0, ax2, 0)
    # align_yaxis(ax, 0, ax3, 0)
    align_yaxis(ax, 0, ax4, 0)
    # ax3.spines['right'].set_position(('outward', 60))
    ax4.spines['right'].set_position(('outward', 70))
    savefig2multiplepaths(fig, savepath=[
        plot_folder + '/Profit_&_Full_Cycles_VS_reff_between_{}_and_{}_{}.png'.format(min(eff), max(eff), price_col)],
                          show=show, tight=tight)


def df_occur(path):
    df = []
    new_labels = []
    for fname in os.listdir(path):
        (fname)
        temp_df = pd.read_csv(path + '/' + fname, index_col='year')
        temp_df.columns = [fname.split('.')[0]]
        label = fname.split('_')[0].replace('-', ' ').title()
        new_labels.append(label)
        df.append(temp_df)
    df_t = pd.concat(df, axis=1)
    df = df_t.loc[1990:2017, :]
    df = df.reset_index()
    data = pd.melt(df, id_vars=['year'], var_name='variable', value_name='value')
    x = sns.barplot(x='year', y='value', hue='variable', data=data)
    x.set_xlabel('Year')
    x.set_ylabel('Occurrences')
    x.set_xticklabels(df.year, rotation=45)
    # plt.title('Occurrences of different batteries keywords over time')
    # title
    plt.grid()
    new_title = 'Battery Types'
    x.legend_.set_title(new_title)
    # replace labels
    for t, l in zip(x.legend_.texts, new_labels):
        t.set_text(l)

    fig = x.get_figure()
    plt.show()
    fig.savefig('occurrences.png')


def soc_in_out_comparison_res(fname, day_str='2017-08-01'):
    soc_cols = ['sto_soc_o', 'sto_soc_c', 'sto_soc']
    power_cols = ['bat_change_o', 'bat_change_c', 'bat_change']
    price_cols = ['Day_Ahead_Auction_o', 'Intraday_Auction_15_minute_call', 'Intraday_Auction_15_minute_call']
    profit_cols = ['profit_o', 'profit_c', 'profit']
    fig, (ax) = plt.subplots(len(price_cols), sharex=True, sharey=True, figsize=[13, 15])

    for num in range(len(price_cols)):
        price_col = fname.split('_subFiles_')[1].replace(".dat", "")
        price_cols.append(price_col)
        if day_str:
            df = readamplres("{0}/{1}".format(DATADIR, fname), unix_column='hoy', drop_zeros=False, drop_str=False).loc[
                 day_str, :]
        else:
            df = readamplres("{0}/{1}".format(DATADIR, fname), unix_column='hoy', drop_zeros=False, drop_str=False)
        c_value = fname.split('_')[5]
        rt_eff = float(fname.split('_')[6])
        df['sto_soc_o'] = df['sto_cap_o'] / 1000
        df['sto_soc_c'] = df['sto_cap_c'] / 1000
        # horizon = filename.split('1h.dat_')[1].split('_subFiles')[0].replace('_', ' - ')

        eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(fname, df, var_col=power_cols[num],
                                                                    profit_col=profit_cols[num])
        #         ax[num].text(0.12, 0.9, s='C-Rate = {}\nProfit = {:.2f} Euro\nEquivalent full cycles = {:.2f}'.format(c_value, gain, eq_fullcycles),
        #                      horizontalalignment='center', multialignment='left', verticalalignment='top', transform=ax[num].transAxes)
        line1 = ax[num].step(df.index, df[soc_cols[num]], where='post', label='State of Charge')
        ax[num].fill_between(df.index, df[soc_cols[num]], step='post', label='State of Charge', alpha=0.5)
        # ax[num].set_yticks(np.arange(0, 1.1, 0.1))
        # ax[num].set_yticklabels(['{:,.0%}'.format(ts) for ts in ax[num].get_yticks()])
        ax1 = ax[num].twinx()
        line2 = ax1.step(df.index, df[price_cols[num]], where='post', color=colorlist[1],
                         label=price_cols[num].replace('_', ' ').title(), alpha=1)
        #         if 'bat_change_c' in df.columns:
        #             print(df['bat_change_c'])
        #             ax2 = ax[num].twinx()
        #             line3 = ax2.step(df.index, df['bat_change_c'], where='post', label='Corrected Power', color=colorlist[2], alpha=0.5)
        #             ax2.fill_between(df.index, df['bat_change_c'], step='post', label='Corrected Power', alpha=0.2)
        #             lines = line1 + line2 + line3
        #         else:
        lines = line1 + line2

        labs = [l.get_label() for l in lines]
        ax[-1].set_xlabel('Time')
        ax[0].set_title('Exemplary Day: {}'.format(day_str))
        ax[num].set_ylabel('SoC in %')
        ax1.set_ylabel('Price in Euro/MWh')
        ax[num].legend(lines, labs, loc='upper right')
        # ax1.set_yticks(np.arange(0, 55, 5))
        ax1.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 1)))

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=pytz.timezone('Europe/Berlin')))
        plt.setp(ax[num].get_xticklabels(), visible=True, rotation='45')
        ax[num].text(0.12, 0.9,
                     s='Round trip efficiency = {:,.0%}\nC-Rate = {}\nProfit = {:.2f} Euro\nEquivalent full cycles = {:.2f}'.format(
                         rt_eff, c_value, gain, eq_fullcycles),
                     horizontalalignment='center', multialignment='left', verticalalignment='top',
                     transform=ax[num].transAxes)
        # ax1.set_yticks(np.arange(0, 60, 5))
        # align_yaxis(ax[num], 0, ax1, 0)
        ax[num].grid(True)

    savefig2multiplepaths(fig, savepath=[plot_folder + '/soc_{}_{}.png'.format(price_cols, day_str)], show=True,
                          tight=True)


def bat_in_out_comparison_res(fname, folder=DATADIR, day_str='01-01-2017'):
    power_cols = ['bat_change_o', 'bat_change_c', 'bat_change']
    price_cols = ['Day_Ahead_Auction_o', 'Intraday_Auction_15_minute_call', 'Intraday_Auction_15_minute_call']
    profit_cols = ['profit_o', 'profit', 'profit_c']
    fig, (ax) = plt.subplots(len(price_cols), sharex=True, sharey=True, figsize=[13, 15])

    for num in range(len(price_cols)):
        price_col = fname.split('_subFiles_')[1].replace(".dat", "")
        price_cols.append(price_col)
        if day_str:
            df = readamplres("{0}/{1}".format(folder, fname), unix_column='hoy', drop_zeros=False, drop_str=False).loc[
                 day_str, :]
        else:
            df = readamplres("{0}/{1}".format(folder, fname), unix_column='hoy', drop_zeros=False, drop_str=False)
        c_value = fname.split('_')[5]
        rt_eff = float(fname.split('_')[6])
        df['sto_soc_o'] = df['sto_cap_o'] / 1000
        df['sto_soc_c'] = df['sto_cap_c'] / 1000

        eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(fname, df, var_col=power_cols[num],
                                                                    profit_col=profit_cols[num])
        line1 = ax[num].step(df.index, df[power_cols[num]], where='post', label='Power')
        ax[num].fill_between(df.index, df[power_cols[num]], step='post', label='Power', alpha=0.5)
        ax1 = ax[num].twinx()
        line2 = ax1.step(df.index, df[price_cols[num]], where='post', color=colorlist[1],
                         label=price_cols[num].replace('_', ' ').title(), alpha=1)
        lines = line1 + line2

        labs = [l.get_label() for l in lines]
        ax[-1].set_xlabel('Time')
        ax[0].set_title('Exemplary Day: {}'.format(day_str))
        ax[num].set_ylabel('Power in kW')
        ax1.set_ylabel('Price in Euro/MWh')
        ax[num].legend(lines, labs, loc='upper right')
        ax1.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 1)))

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=pytz.timezone('Europe/Berlin')))
        plt.setp(ax[num].get_xticklabels(), visible=True, rotation='45')
        ax[num].text(0.18, 0.9,
                     s='Round trip efficiency = {:,.0%}\nC-Rate = {}\nProfit = {:.2f} Euro\nEquivalent full cycles = {:.2f}'.format(
                         rt_eff, c_value, gain, eq_fullcycles),
                     horizontalalignment='center', multialignment='left', verticalalignment='top',
                     transform=ax[num].transAxes)
        # ax1.set_yticks(np.arange(0, 60, 5))
        # align_yaxis(ax[num], 0, ax1, 0)
        ax[num].grid(True)

    savefig2multiplepaths(fig, savepath=[plot_folder + '/{}_{}.png'.format(price_cols, day_str)], show=True, tight=True)


def bat_in_out_comparison_res_2(fname, folder=DATADIR, day_str='01-01-2017'):
    power_cols = ['bat_change_o_o', 'bat_change_c_o', 'bat_change_cc', 'bat_change']
    price_cols = ['Day_Ahead_Auction_o_o', 'Intraday_Auction_15_minute_call',
                  'Intraday_Continuous_15_minutes_Average_Price', 'Intraday_Continuous_15_minutes_Average_Price']
    profit_cols = ['profit_o_o', 'profit_o', 'profit', 'profit_cc']
    fig, (ax) = plt.subplots(len(price_cols), sharex=True, sharey=True, figsize=[13, 15])

    for num in range(len(price_cols)):
        price_col = fname.split('_subFiles_')[1].replace(".dat", "")
        price_cols.append(price_col)
        if day_str:
            df = readamplres("{0}/{1}".format(folder, fname), unix_column='hoy', drop_zeros=False, drop_str=False).loc[
                 day_str, :]
        else:
            df = readamplres("{0}/{1}".format(folder, fname), unix_column='hoy', drop_zeros=False, drop_str=False)
        print(df.head())
        c_value = fname.split('_')[5]
        rt_eff = float(fname.split('_')[6])

        eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(fname, df, var_col=power_cols[num],
                                                                    profit_col=profit_cols[num], combined='3')

        line1 = ax[num].step(df.index, df[power_cols[num]], where='post', label='Power')
        ax[num].fill_between(df.index, df[power_cols[num]], step='post', label='Power', alpha=0.5)
        ax1 = ax[num].twinx()
        line2 = ax1.step(df.index, df[price_cols[num]], where='post', color=colorlist[1],
                         label=price_cols[num].replace('_', ' ').title().replace('O', ''), alpha=1)
        lines = line1 + line2

        labs = [l.get_label() for l in lines]
        ax[-1].set_xlabel('Time')
        ax[0].set_title('Exemplary Day: {}'.format(day_str))
        ax[num].set_ylabel('Power in kW')
        ax1.set_ylabel('Price in Euro/MWh')
        ax[num].legend(lines, labs, loc='lower right')
        ax1.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 1)))

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=pytz.timezone('Europe/Berlin')))
        plt.setp(ax[num].get_xticklabels(), visible=True, rotation='45')
        ax[num].text(0.18, 0.9,
                     s='Round trip efficiency = {:,.0%}\nC-Rate = {}\nProfit = {:.2f} Euro\nEquivalent full cycles = {:.2f}'.format(
                         rt_eff, c_value, gain, eq_fullcycles),
                     horizontalalignment='center', multialignment='left', verticalalignment='top',
                     transform=ax[num].transAxes)
        # ax1.set_yticks(np.arange(0, 60, 5))
        # align_yaxis(ax[num], 0, ax1, 0)
        ax[num].grid(True)

    savefig2multiplepaths(fig, savepath=[plot_folder + '/{}_{}.png'.format(price_cols[-1], day_str)], show=True,
                          tight=True)


def exemplary_day_plot_bat(filename, folder=DATADIR, day_str='01-01-2017', combined='1'):
    fig, (ax) = plt.subplots(1, sharex=True, sharey=True, figsize=FIGSIZE)

    price_col = filename.split('_subFiles_')[1].replace(".dat", "")
    if day_str:
        df = readamplres("{0}/{1}".format(folder, filename), unix_column='hoy', drop_zeros=False, drop_str=False).loc[
             day_str, :]
    else:
        df = readamplres("{0}/{1}".format(folder, filename), unix_column='hoy', drop_zeros=False, drop_str=False)
    c_value = filename.split('_')[5]
    rt_eff = float(filename.split('_')[6])

    # horizon = filename.split('1h.dat_')[1].split('_subFiles')[0].replace('_', ' - ')

    if combined == '1':
        eq_fullcycles, gain = no_eq_full_cycles_and_profit(filename, df)
    elif combined == '2':
        eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(filename, df, var_col='bat_change',
                                                                    profit_col='profit_c')
    elif combined == '3':
        eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(filename, df, var_col='bat_change',
                                                                    profit_col='profit_cc')
    #    ax.text(0.12, 0.9, s='C-Rate = {}\nProfit = {:.2f} Euro\nEquivalent full cycles = {:.2f}'.format(c_value, gain, eq_fullcycles),
    #                 horizontalalignment='center', multialignment='left', verticalalignment='top', transform=ax.transAxes)
    line1 = ax.step(df.index, df['bat_change'], where='post', label='Power')
    ax.fill_between(df.index, df['bat_change'], step='post', label='Power', alpha=0.2, color=colorlist[0])
    # ax[num].set_yticks(np.arange(0, 1.1, 0.1))
    # ax[num].set_yticklabels(['{:,.0%}'.format(ts) for ts in ax[num].get_yticks()])
    ax1 = ax.twinx()
    line2 = ax1.step(df.index, df[price_col], where='post', color=colorlist[1],
                     label=price_col.replace('_', ' ').title(), alpha=1)
    #         if 'bat_change_c' in df.columns:
    #             print(df['bat_change_c'])
    #             ax2 = ax[num].twinx()
    #             line3 = ax2.step(df.index, df['bat_change_c'], where='post', label='Corrected Power', color=colorlist[2], alpha=0.5)
    #             ax2.fill_between(df.index, df['bat_change_c'], step='post', label='Corrected Power', alpha=0.2)
    #             lines = line1 + line2 + line3
    #         else:
    lines = line1 + line2

    labs = [l.get_label() for l in lines]
    ax.set_xlabel('Time')
    ax.set_title('Exemplary Day: {}'.format(day_str))
    ax.set_ylabel('Power in kW')
    ax1.set_ylabel('Price in Euro/MWh')
    ax.legend(lines, labs, loc='lower right')

    ax1.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 1)))

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=pytz.timezone('Europe/Berlin')))
    plt.setp(ax.get_xticklabels(), visible=True, rotation='45')
    ax.text(0.18, 0.9,
            s='Round trip efficiency = {:,.0%}\nC-Rate = {}\nProfit = {:.2f} Euro\nEquivalent full cycles = {:.2f}'.format(
                rt_eff, c_value, gain, eq_fullcycles),
            horizontalalignment='center', multialignment='left', verticalalignment='top', transform=ax.transAxes)
    # ax1.set_yticks(np.arange(0, 60, 5))
    # align_yaxis(ax[num], 0, ax1, 0)
    ax.grid(True)

    savefig2multiplepaths(fig, savepath=[plot_folder + '/{}_{}.png'.format(price_col, day_str)], show=True, tight=True)


def bat_in_out_comparison(*fnames, folder=DATADIR, day_str='01-01-2017', combined='1', single_graph=False):
    fig, (ax) = plt.subplots(len(fnames), sharex=True, sharey=True, figsize=[13, 15])
    price_cols = []

    for num, filename in enumerate(fnames):
        print(len(fnames))
        if single_graph:
            num = 1
        print(num)
        price_col = filename.split('_subFiles_')[1].replace(".dat", "")
        price_cols.append(price_col)
        if day_str:
            df = readamplres("{0}/{1}".format(folder, filename), unix_column='hoy', drop_zeros=False,
                             drop_str=False).loc[day_str, :]
        else:
            df = readamplres("{0}/{1}".format(folder, filename), unix_column='hoy', drop_zeros=False, drop_str=False)
        c_value = filename.split('_')[5]
        rt_eff = float(filename.split('_')[6])

        # horizon = filename.split('1h.dat_')[1].split('_subFiles')[0].replace('_', ' - ')

        if combined == '1':
            eq_fullcycles, gain = no_eq_full_cycles_and_profit(filename, df)
        elif combined == '2':
            eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(filename, df, var_col='bat_change',
                                                                        profit_col='profit_c')
        elif combined == '3':
            eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(filename, df, var_col='bat_change',
                                                                        profit_col='profit_cc')
        #         ax[num].text(0.12, 0.9, s='C-Rate = {}\nProfit = {:.2f} Euro\nEquivalent full cycles = {:.2f}'.format(c_value, gain, eq_fullcycles),
        #                      horizontalalignment='center', multialignment='left', verticalalignment='top', transform=ax[num].transAxes)
        line1 = ax[num].step(df.index, df['bat_change'], where='post', label='Power')
        ax[num].fill_between(df.index, df['bat_change'], step='post', label='Power', alpha=0.2)
        # ax[num].set_yticks(np.arange(0, 1.1, 0.1))
        # ax[num].set_yticklabels(['{:,.0%}'.format(ts) for ts in ax[num].get_yticks()])
        ax1 = ax[num].twinx()
        line2 = ax1.step(df.index, df[price_col], where='post', color=colorlist[1],
                         label=price_col.replace('_', ' ').title(), alpha=1)
        #         if 'bat_change_c' in df.columns:
        #             print(df['bat_change_c'])
        #             ax2 = ax[num].twinx()
        #             line3 = ax2.step(df.index, df['bat_change_c'], where='post', label='Corrected Power', color=colorlist[2], alpha=0.5)
        #             ax2.fill_between(df.index, df['bat_change_c'], step='post', label='Corrected Power', alpha=0.2)
        #             lines = line1 + line2 + line3
        #         else:
        lines = line1 + line2

        labs = [l.get_label() for l in lines]
        ax[-1].set_xlabel('Time')
        ax[0].set_title('Exemplary Day: {}'.format(day_str))
        ax[num].set_ylabel('Power in kW')
        ax1.set_ylabel('Price in Euro/MWh')
        ax[num].legend(lines, labs, loc='upper center')
        # ax1.set_yticks(np.arange(0, 55, 5))
        ax1.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 1)))

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=pytz.timezone('Europe/Berlin')))
        plt.setp(ax[num].get_xticklabels(), visible=True, rotation='45')
        ax[num].text(0.18, 0.9,
                     s='Round trip efficiency = {:,.0%}\nC-Rate = {}\nProfit = {:.2f} Euro\nEquivalent full cycles = {:.2f}'.format(
                         rt_eff, c_value, gain, eq_fullcycles),
                     horizontalalignment='center', multialignment='left', verticalalignment='top',
                     transform=ax[num].transAxes)
        # ax1.set_yticks(np.arange(0, 60, 5))
        # align_yaxis(ax[num], 0, ax1, 0)
        ax[num].grid(True)

    savefig2multiplepaths(fig, savepath=[plot_folder + '/{}_{}.png'.format(price_cols, day_str)], show=True, tight=True)


def soc_price_comparison(*fnames, day_str='01-01-2017', combined='1'):
    fig, (ax) = plt.subplots(len(fnames), sharex=True, sharey=True, figsize=[13, 15])
    price_cols = []

    for num, filename in enumerate(fnames):
        price_col = filename.split('_subFiles_')[1].replace(".dat", "")
        price_cols.append(price_col)
        if day_str:
            df = readamplres("{0}/{1}".format(DATADIR, filename), unix_column='hoy',
                             drop_zeros=False,
                             drop_str=False).loc[day_str, ['sto_soc',
                                                           'bat_in', 'bat_out', 'sto_in',
                                                           'sto_out', 'profit', price_col]]
        else:
            df = readamplres("{0}/{1}".format(DATADIR, filename), unix_column='hoy', drop_zeros=False,
                             drop_str=False).loc[:,
                 ['sto_soc', 'bat_in', 'bat_out', 'sto_in', 'sto_out', 'profit', price_col]]
        c_value = filename.split('_')[5]
        # for index, row in df.iterrows():
        #     if (df.loc[index, 'sto_in'] != 0 and df.loc[index, 'sto_out'] != 0):
        #         if df.loc[index, 'sto_in'] > df.loc[index, 'sto_out']:
        #             df.loc[index, 'sto_in'] = df.loc[index, 'sto_in'] - df.loc[index, 'sto_out']
        #             df.loc[index, 'sto_out'] = 0
        #         elif df.loc[index, 'sto_in'] < df.loc[index, 'sto_out']:
        #             df.loc[index, 'sto_out'] = df.loc[index, 'sto_out'] - df.loc[index, 'sto_in']
        #             df.loc[index, 'sto_in'] = 0
        # df['profit'] = df[price_col] * (df['sto_out'] - df['sto_in']) / 1000
        rt_eff = float(filename.split('_')[6])
        # horizon = filename.split('1h.dat_')[1].split('_subFiles')[0].replace('_', ' - ')

        if combined == '1':
            eq_fullcycles, gain = no_eq_full_cycles_and_profit(filename, df)
        elif combined == '2':
            eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(filename, df, var_col='bat_change',
                                                                        profit_col='profit_c')
        elif combined == '3':
            eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(filename, df, var_col='bat_change',
                                                                        profit_col='profit_cc')  # ax[num].text(0.12, 0.9, s='C-Rate = {}\nProfit = {:.2f} Euro\nEquivalent full cycles = {:.2f}'.format(c_value, gain, eq_fullcycles),
        #                      horizontalalignment='center', multialignment='left', verticalalignment='top', transform=ax[num].transAxes)
        line1 = ax[num].step(df.index, df['sto_soc'], where='post', label='State of Charge')
        ax[num].fill_between(df.index, df['sto_soc'], step='post', label='State of Charge', alpha=0.5)
        ax[num].set_yticks(np.arange(0, 1.1, 0.1))
        ax[num].set_yticklabels(['{:,.0%}'.format(x) for x in ax[num].get_yticks()])
        ax1 = ax[num].twinx()
        line2 = ax1.step(df.index, df[price_col], where='post', color=colorlist[1],
                         label=price_col.replace('_', ' ').title(), alpha=0.5)
        lines = line1 + line2
        labs = [l.get_label() for l in lines]
        ax[-1].set_xlabel('Time')
        ax[0].set_title('Exemplary Day: {}'.format(day_str))
        ax[num].set_ylabel('State of Charge')
        ax1.set_ylabel('Price in Euro/MWh')
        ax[num].legend(lines, labs, loc='upper right')
        # ax1.set_yticks(np.arange(0, 55, 5))
        ax1.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 1)))

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=pytz.timezone('Europe/Berlin')))
        plt.setp(ax[num].get_xticklabels(), visible=True, rotation='45')
        ax[num].text(0.12, 0.9,
                     s='Round trip efficiency = {:,.0%}\nC-Rate = {}\nProfit = {:.2f} Euro\nEquivalent full cycles = {:.2f}'.format(
                         rt_eff, c_value, gain, eq_fullcycles),
                     horizontalalignment='center', multialignment='left', verticalalignment='top',
                     transform=ax[num].transAxes)
        # ax1.set_yticks(np.arange(0, 60, 5))
        # align_yaxis(ax[num], 0, ax1, 0)
        ax[num].grid(True)

    savefig2multiplepaths(fig, savepath=[plot_folder + '/{}_{}.png'.format(price_cols, day_str)], show=True, tight=True)


def ampl_carpet_plot(*fnames, folder=DATADIR, col='sto_soc', combined='1', show=True, figsize=FIGSIZE_carpet,
                     tight=True):
    for filename in fnames:
        df = readamplres("{0}/{1}".format(folder, filename), unix_column='hoy', drop_zeros=True, drop_str=True)
        price_col = filename.split('_subFiles_')[1].replace(".dat", "")
        sampling_step_width_sec = int(df.index.to_series().diff().dt.seconds.mean())
        first_timestamp, last_timestamp = df.index.min().date(), df.index.max().date()
        if combined == '1':
            eq_fullcycles, gain = no_eq_full_cycles_and_profit(filename, df)
        elif combined == '2':
            eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(filename, df, var_col='bat_change',
                                                                        profit_col='profit_c')
        elif combined == '3':
            eq_fullcycles, gain = no_eq_full_cycles_and_profit_combined(filename, df, var_col='bat_change',
                                                                        profit_col='profit_cc')
        print('The optimization data for the file\n{}\n as follows\n\n'.format(filename))
        print('The time series of the operation for the output file\n{}\n\n'.format(df.head()))
        print('The total profit between\n{0} and {1}\nequals to {2} Euro\n\n'.
              format(first_timestamp, last_timestamp, gain.round(5)))
        print('The total full cycles between\n{0} and {1}\nequals to {2} Cycles\n\n'.
              format(first_timestamp, last_timestamp, eq_fullcycles))
        carpet(df[col], zlabel=r'State of Charge', sampling_step_width_sec=sampling_step_width_sec,
               grid=False, figsize=figsize, show=show, tight=tight, perc=True,
               savepath=[plot_folder + '/carpet_plot_soc_from_{0}_to_{1}_with_{2}_cap_{3}_crate_{4}_timeres_{5}.png'.
               format(first_timestamp, last_timestamp,
                      filename.split('_')[2],
                      filename.split('_')[5],
                      filename.split('_')[12].replace(".dat", ""), price_col)])


def linear_reg_price(*fnames, show=True, figsize=FIGSIZE, tight=True):
    for filename in fnames:
        df = read_dat_to_timeseries('{}/{}'.format(input_folder, filename), delimiter=';\t',
                                    comment_sign='#', timecol='unixtimestamp')
        print(df.head())
        print(df.info())
        print(df.describe())
        exit()
        price_cols = extract_price_cols(df)
        var_cols = ['Wind', 'Load', 'Solar', 'Import_Balance', 'Conventional_>_100_MW'] + price_cols
        for price_col in price_cols:
            df = df.loc[:, var_cols]
            for var in var_cols:
                eq_par = np.polyfit(df[var], df[price_col], 1)
                eq = np.poly1d(eq_par)
                x = np.arange(50)
                y = eq(x)
                fig, ax = plt.subplots(1, figsize=figsize)

                ax.scatter(df[var], df[price_col], marker='ts', color=colorlist[2],
                        alpha=0.2)
                ax.plot(x, y, linestyle='--', marker='o', color=colorlist[1])
                # ax.grid()
                ax.set_xlabel('{} in GW'.format(var))
                ax.set_ylabel('Price in Euro/MWh')
                plt.text(x=0.02, y=0.25,
                         s='Regression Equation: {}'.format(eq),
                         backgroundcolor='white',
                         horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
                savefig2multiplepaths(fig,
                                      savepath=[
                                          plot_folder + '/Linear_Reg_{}_{}_in_{}.png'.format(var, price_col,
                                                                                             df.index.year[
                                                                                                 0])],
                                      show=show, tight=tight)


def heatmap_market(*fnames, show=True, figsize=FIGSIZE, tight=True):
    for filename in fnames:
        df = read_dat_to_timeseries('{}/{}'.format(input_folder, filename), delimiter=';\t',
                                    comment_sign='#', timecol='unixtimestamp')
        price_cols = extract_price_cols(df)
        for price_col in price_cols:
            fig, ax1 = plt.subplots(1, figsize=figsize)
            var_cols = ['Import_Balance', 'Conventional_>_100_MW', 'Wind', 'Solar', 'Load'] + [price_col]
            df = df.loc[:, var_cols]
            df.rename(columns={"Import_Balance": "Import", "Conventional_>_100_MW": "Conventional",
                               price_col: 'Price'},
                      inplace=True)
            sns.heatmap(df.corr(), cmap='RdYlGn', ax=ax1, square='auto', vmin=-1, vmax=1)
            savefig2multiplepaths(fig,
                                  savepath=[
                                      plot_folder + '/price_heatmap_{}_in_{}.png'.format(price_col, df.index.year[0])],
                                  show=show, tight=tight)


def carpet_plot_eex(*fnames, year=None, show=True, figsize=FIGSIZE, tight=True):
    for filename in fnames:
        df = read_dat_to_timeseries('{}/{}'.format(input_folder, filename), delimiter=';\t',
                                    comment_sign='#', timecol='unixtimestamp')
        sampling_step_width_sec = int(df.index.to_series().diff().dt.seconds.mean())
        price_cols = extract_price_cols(df)
        for pricecolumn in price_cols:
            if year:
                carpet(df.loc[str(year), pricecolumn], zlabel='Price in Euro/MWh', perc=False, grid=False,
                       carpet_scale_limits=[-300, 300], cmap='jet',
                       figsize=figsize, show=show, tight=tight, sampling_step_width_sec=sampling_step_width_sec,
                       savepath=[plot_folder + '/carpet_plot_{0}_in_{1}.png'.format(pricecolumn, year)])
            else:
                carpet(df.loc[:, pricecolumn], zlabel='Price in Euro/MWh', perc=False, grid=False,
                       carpet_scale_limits=[-300, 300], cmap='jet',
                       figsize=figsize, show=show, tight=tight, sampling_step_width_sec=sampling_step_width_sec,
                       savepath=[plot_folder + '/carpet_plot_{0}_in_{1}.png'.format(pricecolumn, year)])


def prices_boxplot(filename, price_col=None, show=True, figsize=FIGSIZE, tight=True):
    """Plotting """
    df = read_dat_to_timeseries('{}/{}'.format(input_folder, filename), delimiter=';\t',
                                comment_sign='#', timecol='unixtimestamp')
    df = syntools.read_synpro_as_timeseries('{}/{}'.format(input_folder, filename))
    if not price_col:
        price_cols = extract_price_cols(df)
    for price_col in price_cols:
        fig, ax = plt.subplots(1, figsize=figsize)
        df['Month'] = df.index.strftime('%b')
        sns.boxplot(x='Month', y=price_col, data=df, ax=ax)
        # sns.swarmplot(ts='Month', y="Day_Ahead_Auction", data=df, ax=ax, color=".25")
        ax.set_ylabel('Price in Euro/MWh')


        savefig2multiplepaths(fig,
                              savepath=[
                                  plot_folder + '/price_boxplot_{}_in_{}.png'.format(price_col, df.index.year[0])],
                              show=show, tight=tight)


def exemplary_price_day_plot(filename, price_col=None, date_win='15-01-2017', date_sum='15-07-2017', show=True,
                             figsize=FIGSIZE, tight=True):
    df = pd.read_csv(input_folder + '/' + filename, sep=';\t', engine='python', comment='#')
    df = UNIXDataFrameColumn2DataIndex(df, 'unixtimestamp')

    if not price_col:
        price_cols = extract_price_cols(df)

    for price_col in price_cols:
        df.apply(pd.to_numeric, errors='coerce')
        df_win = df[date_win]
        df_sum = df[date_sum]
        fig, ax = plt.subplots(1, figsize=figsize)

        line1 = ax.step(df_win.index, df_win[price_col], where='post',
                        label='Winter', alpha=1, color=colorlist[0])

        line2 = ax.step(df_win.index, df_sum[price_col], where='post',
                        label='Summer', alpha=1, color=colorlist[1])

        ax.fill_between(df_win.index, df_win[price_col], step='post',
                        label='Winter', alpha=0.2, color=colorlist[2])

        ax.fill_between(df_win.index, df_sum[price_col], step='post',
                        label='Summer', alpha=0.2, color=colorlist[3])
        # ax.legend(loc='upper left')
        lines = line1 + line2
        labs = [l.get_label() for l in lines]
        ax.legend(lines, labs, loc='upper center')

        ax.set_xlabel('Time')
        ax.set_ylabel('Price in Euro/MWh')

        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 1)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=pytz.timezone('Europe/Berlin')))
        # plt.setp(ax.get_xticklabels(), visible=True, rotation='45')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, horizontalalignment='center')
        ax.set_yticks(np.arange(-100, 100, 10))
        ax.set_ylim([-20, 80])
        ax.grid()
        plt.text(x=0.02, y=0.95,
                 s='Winter Day: {}\nSummer Day: {}'.format(date_win, date_sum),
                 backgroundcolor='white',
                 horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
        savefig2multiplepaths(fig,
                              savepath=[plot_folder + '/Price_Curve_Example_of_{}_steps_on_{}_{}.png'.format(price_col,
                                                                                                             date_sum,
                                                                                                             date_win)],
                              show=show, tight=tight)


def prices_comparison(*fnames, price_col=None, time_res=None, steps=False, show=True, figsize=FIGSIZE, tight=True):
    df_list = []
    for filename in fnames:
        temp_df = pd.read_csv(input_folder + '/' + filename, sep=';\t', engine='python', comment='#')
        temp_df = UNIXDataFrameColumn2DataIndex(temp_df, 'unixtimestamp')
        if price_col:
            price_cols = [price_col]
        else:
            price_cols = extract_price_cols(temp_df)
        temp_df = temp_df.loc[:, price_cols]
        cols = temp_df.columns
        temp_df[cols] = temp_df[cols].apply(pd.to_numeric, errors='coerce')
        if time_res:
            temp_df = temp_df.resample(time_res).mean()
        else:
            pass
        # print(temp_df.head())
        # exit()
        df_list.append(temp_df)
    df = pd.concat(df_list, axis=1)
    fig, ax = plt.subplots(1, figsize=figsize)
    lines = []
    for num, column in enumerate(df.columns):
        if steps:
            line = ax.step(df.index, df[column], where='post', label=column, color=colorlist[num])
        else:
            line = ax.plot(df.index, df[column], label=column, color=colorlist[num])
        lines += line
    print(df.head())
    labs = [l.get_label() for l in lines]
    ax.legend(lines, labs, loc='lower right')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price in Euro/MWh')
    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter('%b')
    # ax.set_yticks(np.arange(0, 110, 10))
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)
    # ax.set_xticks(df.index.values)
    # ax.set_yticks(np.arange(0, 100, 10))
    ax.grid(True)
    ax.text(0.15, 0.8, s='Resample Window: {}\nYear: {}'.format(time_res, df.index.year[0]),
            horizontalalignment='center', multialignment='left', verticalalignment='top', transform=ax.transAxes)
    savefig2multiplepaths(fig,
                          savepath=[
                              plot_folder + '/Prices_Comparison_in_{}_steps_{}.png'.format(df.index.year[0], steps)],
                          show=show, tight=tight)


def plot_rollingmeaneex_yearly(*fnames, figsize=FIGSIZE):
    """Plot EEX-Price curve for the mean day of a defined period"""
    days_dict = {'Original Price': 0, 'Moving Daily Average': 1, 'Moving Weekly Average': 7,
                 'Moving Monthly Average': 30}  # Number of days in every moving average you need to calculate
    for filename in fnames:
        pd_ts = pd.read_csv(input_folder + '/' + filename, sep=';\t', engine='python', comment='#')
        pd_ts = UNIXDataFrameColumn2DataIndex(pd_ts, 'unixtimestamp')
        price_cols = extract_price_cols(pd_ts)
        pd_ts = pd_ts.loc[:, price_cols]
        cols = pd_ts.columns
        pd_ts[cols] = pd_ts[cols].apply(pd.to_numeric, errors='coerce')
        for column in pd_ts:
            fig, ax = plt.subplots(1, figsize=figsize)
            str_first_day = str(pd_ts.index[1])[:]
            for key, value in days_dict.items():
                if value == 0:
                    rmean = pd_ts.loc[str_first_day:, column]
                else:
                    rmean = pd_ts[column].rolling(24 * value).mean()[str_first_day:]
                ax.plot(rmean, label=key)
            ax.set_ylabel('Price in Euro/MWh')
            # ax.set_title('{}'.format(column.replace('_', ' ')))
            locator = mdates.MonthLocator()
            fmt = mdates.DateFormatter('%b')
            # ax.set_yticks(np.arange(0, 110, 10))
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(fmt)
            ax.set_xlabel('Date')
            ax.legend(loc='upper center')
            ax.set_yticks(np.arange(-100, 250, 25))
            # ax.text(0.5, 0.95, s='Market: {}\nYear: {}'.format(column.replace('_', ' '), pd_ts.index.year[0]),
            #         backgroundcolor='white',
            #         horizontalalignment='center', multialignment='center', verticalalignment='top',
            #         transform=ax.transAxes)
            ax.grid(True)
            savefig2multiplepaths(fig, tight=True, show=True,
                                  savepath=[
                                      plot_folder + '/Prices_rollingmeaneex_in_{}_steps_{}.png'.format(
                                          pd_ts.index.year[0],
                                          column)])


def missing_dates(df):
    nan = df.resample('1H', base=df.index[0].second).mean()
    nan_no = nan.isnull().sum().sum()
    nan_rows = nan[nan.isnull().T.any()].index.values.tolist()
    print('The total number of missing Timestamps: {}\nThe missing rows:\n{}'.format(nan_no, nan_rows))
    date_set = set(df.index[0] + timedelta(x) for x in range((df.index[-1] - df.index[0]).days))
    missing = sorted(date_set - set(df.index))
    print('The missing dates:\n{}'.format(missing))


def rename_files_in_folder(path, old_str, new_str):
    files = os.listdir(path)
    for file in files:
        if file.startswith('res') and file.endswith('.dat'):
            new_name = file.replace(old_str, new_str)
            os.rename(os.path.join(path, file), os.path.join(path, new_name))
            print('The file {} is converted to {}'.format(os.path.join(path, file), os.path.join(path, new_name)))


def duplicate_files_with_new_name(path):
    files = os.listdir(path)
    array = np.arange(0.5, 40.5, 0.5)
    old_str = 'res_0_1000_0_0_1_1_0_energycharts_downloads_price_2017_en_1h.dat_24.0_12.0_H_subFiles_Day_Ahead_Auction.dat'
    for file in files:
        for item in array:
            new_str = old_str.replace('0_energycharts', '{}_energycharts'.format(str(item)))
            shutil.copy(path + '/' + old_str, path + '/' + new_str)
            print('The file {} is converted to {}'.format((path + '/' + file, path + '/' + new_str)))


def profit_value_change(*args):
    for item in args:
        for file in os.listdir(item):
            df = pd.read_csv(item + '/' + file, sep=',', index_col='Unnamed: 0')
            cols = ['bat_in', 'bat_out']
            df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
            price_col = file.split('_subFiles_')[1].replace(".dat", "")
            rt_eff = float(file.split('_')[6])
            time_res = file.split('_')[12].replace('.dat', '')
            if time_res == '1h':
                step = 1
            elif time_res == '30min':
                step = 0.5
            elif time_res == '15min':
                step = 0.25
            df['profit'] = df[price_col] * (df['bat_out'] - df['bat_in']) * step / 1000
            df.to_csv('{}/{}'.format(item, file), sep=',')
            print('Done!\n')


def new_profit_x(filename):
    price_col = filename.split('_subFiles_')[1].replace(".dat", "")
    df = readamplres("{0}/{1}".format(DATADIR, filename), unix_column='hoy', drop_zeros=False, drop_str=True)
    time_res = filename.split('_')[12].replace('.dat', '')
    if time_res == '1h':
        step = 1
    elif time_res == '30min':
        step = 0.5
    elif time_res == '15min':
        step = 0.25
    df['energy_profit'] = df[price_col] * (df['sto_out'] - df['sto_in']) / 1000
    df['power_profit'] = df[price_col] * (df['bat_out'] - df['bat_in']) * step / 1000
    gain = df['profit'].sum()
    energy_gain = df['energy_profit'].sum()
    power_gain = df['power_profit'].sum()
    print(df[['profit', 'energy_profit', 'power_profit']].head(10))
    print('\n\nOriginal Profit: {}\nEnergy Profit: {}\nPower Profit: {}'.format(gain, energy_gain, power_gain))


def plotBarAnalysis(restable=None, columnname='chp_motor_onCHP1', figsize=[12, 10],
                    savepath=None, bw=False):
    """
    NOT FINISHED!
    Plot horizontal barplot of the chp runtime for different szenarios.
    To do: Read in the correct times

    :param restable:    results table as given out by resanalysis.getsummedenergy()
    :param columnname:    column to be plotted
    :param figsize:        figsize in format (width,height)
    :param savepath:    savepath
    """

    ind = np.arange(4)
    width = 1
    tick_pos = ind  # [pos + width / 2 for pos in ind]

    CONST = 3875  # restable[columnname]['constantPrice']
    EEX = 3774  # restable[columnname]['priceEurMWh']
    HTNT = 3857  # restable[columnname]['HTNT']
    HTNT_STAT = 3719

    if bw:
        colorlist = [(0.3, 0.3, 0.3), (0.5, 0.5, 0.5), (0.7, 0.7, 0.7)]

    plt.close("all")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=False, sharey=False, figsize=figsize)

    # LEFT: RUNTIME
    ax1.bar(ind[0], CONST, width, color=colorlist[0])
    ax1.text(tick_pos[0], CONST * 1.02, "{} h".format(CONST), fontsize=21, fontname='Ubuntu',
             fontweight='normal', horizontalalignment='center', verticalalignment='center')

    # MID: #CHP START
    #     ax1.bar(ind[1], EEX, width, color=colorlist[1])
    #     ax1.text(tick_pos[1], EEX * 1.02, "{} h".format(EEX), fontsize=21, fontname='Ubuntu',
    #              fontweight='normal', horizontalalignment='center', verticalalignment='center')
    # RIGHT: STORAGE LOSSES
    #     ax1.bar(ind[2], HTNT, width, color=colorlist[2])
    #     ax1.text(tick_pos[2], HTNT * 1.02, "{} h".format(HTNT), fontsize=21, fontname='Ubuntu',
    #              fontweight='normal', horizontalalignment='center', verticalalignment='center')

    # addmanuallegend(ax1, ['EEX', 'HT-LT', 'Constant'], alphas=[1, 1, 1],  # , colors=[colorlist[0], colorlist[1]]
    #                     edgecolors=['white', 'white', 'white'], ncol=1, loc=1, facealpha=0.95)

    #    plt.setp(ax1.get_yticklabels(), visible=False)
    ticks_label = ['Constant Price', 'EEX Day Ahead Price', 'Dynamic HT/LT tariff']
    plt.legend(loc='best', framealpha=0.5, fancybox=True)
    plt.xticks(tick_pos, ticks_label, rotation=0)
    plt.xlim([min(tick_pos) - width * 1, max(tick_pos) + width * 1])

    plt.ylim(0, 5000)

    ax1.set_xlabel(r'Simulated Scenarios')
    ax1.set_ylabel('CHP runtime [h/year]')

    savefig2multiplepaths(fig, savepath, show=True)


if __name__ == '__main__':
    filename1 = 'res_0_1000_0_0_0.5_1_energycharts_downloads_price_2017_en_1h.dat_365.0_0.0_D_subFiles_Day_Ahead_Auction.dat'
    filename2 = 'res_0_1000_0_0_0.5_1_energycharts_downloads_price_2017_en_15min.dat_365.0_0.0_D_subFiles_Intraday_Auction_15_minute_call.dat'
    optimizing_markets(filename1, filename2)
    exit()
