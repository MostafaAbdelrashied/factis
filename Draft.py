'''
Created on 07.05.2018
For fun
@author: mabdelra
'''
import os
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import scipy
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier

from create_ampl_input import extract_price_cols
from genpy import ies_style
from genpy import synpro_tools as spt
from genpy.datacrunching import read_dat_to_timeseries, UNIXDataFrameColumn2DataIndex
from genpy.seaborn_plotting import IES_SearbornWrapper
from paths import input_folder
from plot_framework import readcsvfile
from plottools import savefig2multiplepaths

ies_style.set_layout(style='a4arial')
iescolors = ies_style.ISEColors
colorlist = iescolors.get_colorlist()
IES_SearbornWrapper().get_ies_seaborn()


# Gather our code in a main() function

def reading_csv_file(filename, price_cols=False):
    df = read_dat_to_timeseries('{}/{}'.format(input_folder, filename), delimiter=';\t',
                                comment_sign='#', timecol='unixtimestamp')
    feature_labels = ['Wind', 'Solar', 'Conventional_>_100_MW', 'Import_Balance', 'Load']
    x = df.loc[:, 'Wind'][-1:]
    print(x.head())
    exit()
    if not price_cols:
        price_cols = extract_price_cols(df)

    training_df = df['2018-01']
    predicition_df = df['2018-02-01']
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(training_df[feature_labels], training_df['Day_Ahead_Auction'])
    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_parms=None, n_jobs=1, n_neighbors=5,
                         p=2, weights='uniform')
    predicition = knn.predict(predicition_df)
    print(predicition)


def plot():
    x = np.arange(0, 20, 1)
    y = 108000 * (1 - 0.12) ** x
    z = 23722 * (1 + 0.007) ** x
    fig, ax = plt.subplots(1)
    ax.plot(x + 2018, y, linestyle='--', marker='o', color=colorlist[0], label='LCC')
    ax.plot(x + 2018, z, linestyle='--', marker='o', color=colorlist[1], label='Profit')
    ax.set_xlabel('Year')
    ax.set_ylabel('Value in Euro/yr')
    plt.grid()
    ax.set_xticks(np.arange(0, 20, 1) + 2018)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, horizontalalignment='center')

    plt.legend()
    # plt.text(ts=0.5, y=0.9, wrap=True,
    #          s='Logarithmic Regression Equation\n{}'.format(r'$L_n = L_0 * (1 - \frac{p}{100})^N$'),
    #          backgroundcolor='white',
    #          horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
    plt.show()


def func2():
    x = np.arange(0, 20, 1)
    y = 108000 * (1 - 0.12) ** x
    z = 23722 * (1 + 0.007) ** x

    y2 = 108000 * (1 - 0.12 * x)
    z2 = 23722 * (1 + 0.007 * x)

    y3 = 108000 * (2.718281828) ** (-0.12 * x)
    z3 = 23722 * (2.718281828) ** (0.007 * x)

    fig, ax = plt.subplots(1)

    ax.plot(x, y, linestyle='--', marker='o', color=colorlist[0], label='Logarithmic Regression - LCC')
    ax.plot(x, z, linestyle='--', marker='o', color=colorlist[2], label='Logarithmic Regression - Profit')

    # ax.plot(ts,y2, linestyle='-.', marker='ts', color=colorlist[1], label = 'Linear Regression - LCC')
    # ax.plot(ts,z2, linestyle='-.', marker='ts', color=colorlist[3], label = 'Linear Regression - Profit')

    ax.plot(x, y3, linestyle=':', marker='*', color=colorlist[4], label='Future Value - LCC')
    ax.plot(x, z3, linestyle=':', marker='*', color=colorlist[5], label='Future Value - Profit')

    ax.grid()
    # ax.set_xticks(np.arange(0, 20, 1) + 2018)
    # plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, horizontalalignment='center')

    ax.set_xlabel('Year')
    ax.set_ylabel(r'$L_n$')
    plt.legend()
    # $L_n = L_0 * (\frac{100 - p * N}{100})$
    # $L_n = L_0 * (\frac{100 - p}{100})^N$
    # $L_n = L_0 * e^{\frac{p}{100} * N}$
    # plt.text(ts=0.02, y=0.4, wrap=True,
    #          s='Logarithmic Regression Equation: {}\nLinear Regression Equation: {}\nAnnual Growth Rate: {}'.format(r'$L_n = L_0 * (\frac{100 - p}{100})^N$',
    #                                                                                         r'$L_n = L_0 * (\frac{100 - p * N}{100})$',
    #                                                                                         r'$L_n = L_0 * e^{\frac{p}{100} * N}$'),
    #          backgroundcolor='white',
    #          horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    plt.show()


def evaluation_matrix(actual_df, predic_df):
    x = actual_df['p_el']
    y = predic_df['p_el']
    pearson_corr = pearsonr(x, y)
    print("Pearson's correlation coefficient:\n{}\n***************************\n".format(pearson_corr))
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    coefficient_of_dermination = r_value ** 2
    print("Coefficient of determination R^2:\n{}\n***************************\n".format(coefficient_of_dermination))
    mse = mean_squared_error(x, y)
    rmse = sqrt(mse)  # ts = actual values, y = predicted values
    print("Root mean square error RMSE:\n{}\n***************************\n".format(rmse))
    x, y = np.array(x), np.array(y)
    mape = np.mean(np.abs((x - y) / x)) * 100
    print("Mean abosolute percentage error MAPE:\n{}\n***************************\n".format(mape))


def new_func(fname):
    df_main = readcsvfile(fname)
    wind_df = df_main.loc[:, ['Wind', 'unixtimestamp', 'Day_Ahead_Auction']]
    eq_par = np.polyfit(wind_df.Wind, wind_df.Day_Ahead_Auction, 1)
    eq = np.poly1d(eq_par)
    x = np.arange(50)
    y = eq(x)
    fig, ax = plt.subplots(1)

    ax.plot(wind_df.Wind, wind_df.Day_Ahead_Auction, linestyle='None', marker='ts', color=colorlist[2], alpha=0.2)
    ax.plot(x, y, linestyle='--', marker='o', color=colorlist[1])
    ax.grid()
    ax.set_xlabel('Wind in GW')
    ax.set_ylabel('Price in Euro/MWh')
    plt.text(x=0.02, y=0.25,
             s='Regression Equation: {}'.format(eq),
             backgroundcolor='white',
             horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    plt.show()


class Multilingual:
    """Builds object where each shortname is represented by multiple longnames (e.g. in different laguages).
    Paramas:
     shortnames - List of shortnames
     dictoflanguages - Dict with language as key and list of names as value
    chosenlanguage - key of choosen language

    - Example -
    language='e'
    languagedict = multilingual(['xlabel', 'ylabel'],
                     'e':['price', 'time'],
                     'g':['Preis', 'Zeit'],
                     language)
    print languagedict.chosen['xlabel']

    languagedict.append(['test', 'test2'], {'e':['a', 'b'], 'g':['1', '2']})
    print languagedict.g['test']

    """

    def __init__(self, shortnames, dictoflanguages, chosenlanguage=None, verbose=False):
        self.shortnames = shortnames
        self.dictoflanguages = dictoflanguages
        self.verbose = verbose
        for k, _ in list(self.dictoflanguages.items()):
            setattr(
                self, k, dict(list(zip(self.shortnames, self.dictoflanguages[k]))))
        if chosenlanguage is not None:
            self.chosen = getattr(self, chosenlanguage)
        self.checkconsistency()

    def __str__(self):
        import json
        return json.dumps(self.dictoflanguages, sort_keys=True, indent=4)

    def languages(self):
        for key in list(self.dictoflanguages.items())[0]:
            print(key)

    def checkconsistency(self):
        for k, v in list(self.dictoflanguages.items()):
            if len(v) == len(self.shortnames):
                if self.verbose:
                    print(
                        '{} has the same number of elements as the list of shortnames'.format(k))
            else:
                raise Exception('The number of elements in', k, len(v),
                                'is not equal to the number of shortnames', len(self.shortnames))

    def append(self, newshortnames, newdictoflanguages):
        self.shortnames.extend(newshortnames)
        for k, v in list(newdictoflanguages.items()):
            self.dictoflanguages[k].extend(v)
        self.checkconsistency()


def carpet(data, cols, zlabel, savepath, sampling_step_width_sec=900, show=True, language='e',
           tight=False,
           figsize=None, dpi=300):
    # Select language
    languagedict = Multilingual(['xlabel', 'ylabel'],
                                {'g': [r'Tag des Jahres', r'Stunde des Tages'],
                                 'e': [r'Day of the Year', 'Hour of the Day']},
                                chosenlanguage=language)
    languagedict = languagedict.chosen
    ############################################
    # Look for the number of samples per hour
    samples_per_hour = int(3600.0 / sampling_step_width_sec)
    fig, (ax) = plt.subplots(nrows=
                             len(cols), ncols=1, sharex=True, sharey=True, figsize=figsize)
    for i in range(len(cols)):
        arr = data.iloc[:, i].values
        last_good_index = int(
            (len(arr) - (np.mod(len(arr), samples_per_hour * 24))))

        arr = arr[:last_good_index]

        carpet_data = np.reshape(arr, (samples_per_hour * 24, -1), order='F')
        im = ax[i].imshow(carpet_data, aspect='auto', cmap=plt.get_cmap("jet"))
        ax[i].set_ylabel(languagedict['ylabel'])
        ax[i].set_title(cols[i])
        cb = plt.colorbar(im, ax=ax[i])
        cb.set_label(zlabel)

    plt.xlabel(languagedict['xlabel'])
    plt.yticks([w * int(samples_per_hour)
                for w in range(0, 24, 4)], [w for w in range(0, 24, 4)])
    savefig2multiplepaths(
        fig, savepath=savepath, show=show, tight=tight, dpi=dpi)
    return (fig, ax)


# the program.
if __name__ == '__main__':
    synGHD_dir = 'synGHD_el_service_office_el.csv'
    measured_dir = 'measured_el_umwelt_22_MWh.csv'
    slp_dir = 'SLP_el_G1_24_MWh_gas_None_0_MWh.csv'
    output_dir = os.getcwd()
    # read profiles separately
    synGHD_W = spt.read_synpro_as_timeseries(synGHD_dir)
    SLP_G1_kW = spt.read_synpro_as_timeseries(slp_dir)
    measured_kW = spt.read_synpro_as_timeseries(measured_dir)

    result = pd.merge(synGHD_W, SLP_G1_kW, how='inner', on=['YYYYMMDD', 'hhmmss', 'unixtimestamp'])
    ts = pd.merge(result, measured_kW, how='inner', on=['YYYYMMDD', 'hhmmss', 'unixtimestamp'])
    ts = UNIXDataFrameColumn2DataIndex(ts, unix_column='unixtimestamp', timezone=pytz.timezone('Europe/Berlin'))
    ts.rename(columns={'p_el_x': 'synGHD', 'P_el': 'SLP', 'p_el_y': 'Measured'}, inplace=True)
    ts.drop(['Qflow_gas', 'YYYYMMDD', 'hhmmss', 'unixtimestamp'], axis=1, inplace=True)
    scale_factor_synGHD = 1.4
    ts["synGHD"] = ts["synGHD"] / (scale_factor_synGHD)  # W->kW
    scale_factor_SLP = 0.45
    ts["SLP"] = ts["SLP"] * scale_factor_SLP
    # settings of plots:
    ise_colors = ["#179C7D", "#EB6A0A", "#B1C800", "#462915", "#33B8CA",
                  "#E21A00", "#9999FF", "#006E92", "#FFD700", "#666666", "#8B4513"]

    FIGWIDTH = 6.3 * 2
    FIGHIGHT = FIGWIDTH / 3
    FIGSIZE = [FIGWIDTH, FIGHIGHT]
    ts["synGHD"]['2017-03-26 02:00:00':'2017-10-29 03:00:00'] = ts["synGHD"][
                                                                '2017-03-26 01:00:00':'2017-10-29 02:00:00']
    cols = ['synGHD', 'SLP', 'Measured']
    carpet(ts, ts.columns, zlabel=r'Power', savepath=output_dir, figsize=FIGSIZE)
    exit()
