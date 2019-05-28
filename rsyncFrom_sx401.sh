#!/bin/bash

rsync -v -r -t -u sx401:/net/home/o/olutz/amplopt/output/generic/generic_out.dat output/generic/




########################################
# Call a shell script from python code #
#                                      #
# >>> import subprocess                #
# >>> subprocess.call(['./test.sh'])   #
########################################

########################################################################################################################
## OPTIONS (see https://wiki.ubuntuusers.de/rsync/)
########################################################################################################################
# -v increase verbosity
# -r recurse into directories
# -t keep times of the original file
# -u update only modified files
########################################################################################################################