import os

import matplotlib.dates as mdates
import pytz
from matplotlib import pyplot as plt

from datacrunching import UNIXDataFrameColumn2DataIndex
from genpy import synpro_tools as spt
from genpy import timeseries_tools as tst
from plottools import carpet

import pandas as pd

# directories
# synGHD_dir = "D:/synGHD/validation_el/synGHD_profiles/synGHD_el_1_1_office_3_el_75_MWh.csv"
synGHD_dir = "synGHD_el_service_office_el.csv"
SLP_G1_dir = "SLP_el_G1_24_MWh_gas_None_0_MWh.csv"
measured_dir = "measured_el_umwelt_22_MWh.csv"
output_dir = os.getcwd()
# read profiles separately
synGHD_W = spt.read_synpro_as_timeseries(synGHD_dir)
SLP_G1_kW = spt.read_synpro_as_timeseries(SLP_G1_dir)
measured_kW = spt.read_synpro_as_timeseries(measured_dir)

result = pd.merge(synGHD_W, SLP_G1_kW, how='inner', on=['YYYYMMDD', 'hhmmss', 'unixtimestamp'])
ts = pd.merge(result, measured_kW, how='inner', on=['YYYYMMDD', 'hhmmss', 'unixtimestamp'])
ts = UNIXDataFrameColumn2DataIndex(ts, unix_column='unixtimestamp', timezone = pytz.timezone('Europe/Berlin'))
ts.rename(columns={'p_el_x': 'synGHD', 'P_el': 'SLP', 'p_el_y': 'Measured'}, inplace=True)
ts.drop(['Qflow_gas', 'YYYYMMDD', 'hhmmss', 'unixtimestamp'], axis = 1, inplace=True)
# time_shift_df = ts['2017-03-26 01:00:00':'2017-10-29 02:00:00']
# time_shift_df.index = time_shift_df.index.shift(1,'H')
# ts['2017-03-26 01:00:00':'2017-10-29 02:00:00'] = time_shift_df.index
# print(time_shift_df)
# print(ts['2017-03-26 01:00:00':'2017-03-26 04:00:00'])
# create the validation ts
scale_factor_synGHD = 1.4
ts["synGHD"] = ts["synGHD"] / (scale_factor_synGHD)  # W->kW
scale_factor_SLP = 0.45
ts["SLP"] = ts["SLP"] * scale_factor_SLP
# settings of plots:
ise_colors = ["#179C7D", "#EB6A0A", "#B1C800", "#462915", "#33B8CA",
              "#E21A00", "#9999FF", "#006E92", "#FFD700", "#666666", "#8B4513"]

FIGSIZE_carpet = [16, 16 / 3]
ts["synGHD"]['2017-03-26 02:00:00':'2017-10-29 03:00:00'] = ts["synGHD"]['2017-03-26 01:00:00':'2017-10-29 02:00:00']
for device in ts.columns:
    tst.carpets(ts=ts,
                names=[device],
                # create_y_label(english=english, scale=scale, Wh_or_W="W"),
                zlabel="Electrical Load in kW",
                imfilename=output_dir,
                bw=False,
                carpet_scale_limits=None,
                title=None,
                language="e",  # "e", "g"
                saveas=".png",
                dpi=300,
                legend=True,
                figsize=[6.6, 4.95],
                show=True,
                save=True)