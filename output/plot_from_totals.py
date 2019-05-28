#!/usr/bin/python
##
##Plot defined data from different files
##
import os, glob
import re
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


import sys, inspect, os
cmd_parentfolder = '/'.join((os.path.realpath(os.path.dirname(inspect.getfile( inspect.currentframe() )))).split('/')[:-1])
if cmd_parentfolder not in sys.path:
    sys.path.insert(0, cmd_parentfolder)
from ampl import  UNIXDataFrameColumn2DataIndex, subPartinDF, csv_to_ts, interpret_spec

####-
myFolder =  (os.path.realpath(os.path.dirname(inspect.getfile( inspect.currentframe() ))))

# 

def import_battery_sizes(spec_filename):
    """Reads the files starting with 'Totals_res_' in the folder. 
    Reads the value of the battery size from the filename.
    Creates a list with all the values found.
    """
    list_of_batteries=[]
    for filename in glob.glob('Totals_res*{0}'.format(spec_filename)):
        a = filename.split("_")[2]
        list_of_batteries.append(a)
        
    #sort the list as alphanumerical string, needs to import library re
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    list_of_batteries.sort( key=alphanum_key ) 

    return list_of_batteries
    

def read_values(filename, variable, batteries_list):
    """Uses the list of batteries to create the files
    and creates a list from the values of the variable
    """
    list_of_values = []
    for item in batteries_list :
        #create the file name from the battery data and read the value of the defined variable
        actual_file = 'Totals_res_{0}_{1}'.format(item, filename)
        df = pd.read_csv('{}'.format(actual_file), sep = ' ')
        a = float(getattr(df, variable))
        list_of_values.append(a)

    return list_of_values    



def plot_values(batteries_list, data_list, variable, filename):
    """"Plot the data in bar and line form"""
    # Frame, size and background
    plt.figure(num=None, figsize=(16,8), dpi=200, facecolor='w', edgecolor='k')   
    # Graphs
    positions = plt.bar(xrange(len(batteries_list)), data_list, width=0.7, align = 'center', color=[[0.090,0.612,0.490]]) #colors in RGB as fraction of 255
    plt.plot(data_list, color = 'r', linewidth=3)

    # axes manipulation
    axes = plt.subplot(111)
    axes.spines['right'].set_color('none')
    axes.spines['top'].set_color('none')
    
    axes.xaxis.set_ticks_position('bottom')
    axes.spines['bottom'].set_position(('axes', -0.02))
    axes.spines['bottom'].set_lw(3)
    axes.tick_params(axis='ts', direction='out', width=3, length=7, color='k', labelsize=10, pad=10)
     
    
    axes.yaxis.set_ticks_position('left')
    axes.spines['left'].set_position(('axes', -0.00))
    axes.spines['left'].set_lw(3)
    axes.tick_params(axis='y', direction='out', width=3, length=7, color='k', labelsize=10, pad=10)
    
    axes.get_yaxis().set_major_formatter(FuncFormatter(percentage_formatter)) # Changes decimal value to pecentage value
       
    axes.xaxis.grid(False)
    axes.yaxis.grid(False)
 
    # Title, axes and labels
    plt.title('{0} Vs. Battery Capacity'.format(variable.replace ("_", " ")), fontsize=26).set_y(1.1) #set_y command add space between title and graph
    plt.xlabel('Battery Capacity in kWh', fontsize=20, labelpad=20) #labelpad add space between label and axis
    plt.ylabel('{0} in %'.format(variable.replace ("_", " ")), fontsize=20, labelpad=20) #labelpad add space between label and axis
    plt.xticks(xrange(len(batteries_list)), batteries_list, ha = 'center', fontsize=18)
    
    plt.yticks(fontsize=18)

    # Text box
    PV_Gen = 3.3
    Load = 6.1
    
    textstr = 'PV Generation = {0} MWh/a\nLoad                = {1} MWh/a'.format(PV_Gen, Load)
    props = dict(boxstyle='round', facecolor='w', alpha=0.5)
    plt.text(0.01, 0.99, textstr, transform=axes.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    
    # Display/Save Plot
 
    plt.tight_layout()  ################ adjust modified plot to the size of the figure
    
    
    # Save to file
    plt.savefig('Plot_{0}_{1}.png'.format(variable, filename[:-4]))


def percentage_formatter(x,p):
    """"Function to change axis label from decimal to percentage"""
    percent_symbol = '%'
    return '%d %s' % (x*100, percent_symbol)



if __name__ == "__main__":
    #filename = 'AMPL_2010_SSH_Datensatz_15min.dat'
    specDict = interpret_spec(myFolder)
    spec_filename = '{name}'.format(**specDict)

    
    variable_list = ['Self_Consumption', 'Self_Sufficiency']
    batteries_list=import_battery_sizes(spec_filename)
    for variable in variable_list:    
        data_list=read_values(spec_filename, variable, batteries_list)
        plot_values(batteries_list, data_list, variable, spec_filename)



