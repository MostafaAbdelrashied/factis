#!/usr/bin/python
### Calculate energy totals in kWh and maximum power ir kW from file defined in spec.txt
### 

from os import listdir, path 
from os.path import isfile, join

####+ local/relativ modul import
import sys, inspect, os
cmd_parentfolder = '/'.join((os.path.realpath(os.path.dirname(inspect.getfile( inspect.currentframe() )))).split('/')[:-1])
if cmd_parentfolder not in sys.path:
    sys.path.insert(0, cmd_parentfolder)
from ampl import  UNIXDataFrameColumn2DataIndex, subPartinDF, csv_to_ts, interpret_spec
from paths import input_folder, output_folder, parsed_files_dir, generic_folder, specpath

####-
myFolder =  (os.path.realpath(os.path.dirname(inspect.getfile( inspect.currentframe() ))))

### Functions
def getSortedListofFilesinFolder(path = "."):  # input/AMPL_2012_Referenzdatensatz.dat_subFiles
    """
    creates a sorted list of the files located in the folder (path)
    """
    onlyfiles = [ f for f in listdir(path) if isfile(join(path, f)) ]
    sorted_onlyfiles = sorted(onlyfiles)
    return onlyfiles#sorted_onlyfiles

def combinetotinfolder(path):
    totals = getSortedListofFilesinFolder(path)
    print totals
    with open("{}.csv".format(path), "w") as ctot:
        for total in totals:
            with open("{}/{}".format(path, total), "r") as totfile:
                print "{}/{}".format(path, total)
                iteratabletotfile = totfile.readlines()
                if total == totals[0]:
                    fileinput  = iteratabletotfile[0]
                    fileinput += iteratabletotfile[1]
                else:
                    fileinput += iteratabletotfile[1]
        ctot.write(fileinput)
    return fileinput

if __name__ == "__main__":
    from sys import argv
    folderoftotals = argv[1]
    folderoftotals = folderoftotals.rstrip('\\')
    combinetotinfolder(folderoftotals)




