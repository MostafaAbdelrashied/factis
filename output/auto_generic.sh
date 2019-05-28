#!/bin/bash

# merge the data from the input file and the results of the optimization into one file with a table format.


parsed_path=$1   # merged document
generic_path=$2   # file with the results of the optimization
input_path=$3   # file with the input data
PVscale=$4    # PV_scale
loadscale=$5    #load scale


awk 'BEGIN{FS="["} !/\[0\]/ && /\[/ {print $1, $2}' $generic_path > clean_generic_out1.tmp 	#interprets [ as delimiter and writes all lines without <[0]> but with <[>

#Data
 tr -d ] < clean_generic_out1.tmp > clean_generic_out2.tmp 					# deletes all <]>
  awk '{print $3}' clean_generic_out2.tmp > clean_generic_out3.tmp                               #prints data only file
  T=`awk 'END{print $2}' clean_generic_out2.tmp`
   awk -v t=$T 'ORS=NR%t?" ":"\n"' clean_generic_out3.tmp > clean_generic_out4.tmp 			  #new cloumn every T row
    transpose.sh clean_generic_out4.tmp > clean_generic_out5.tmp 				   # transpond columns 2 rows

#Header
 gawk '{print $1}' clean_generic_out1.tmp > names1.tmp 						# writes first row only 
  uniq names1.tmp > names2.tmp 									 #deletes all redundancy
   transpose.sh names2.tmp > names3.tmp								  #transpsoes column 2 row

#Merge Header with data											
    cat names3.tmp > variables.dat							           #writes header in final file 
     cat clean_generic_out5.tmp >> variables.dat						   # writes data under header

#Build input in data format 

cut -d" " -f 2- $input_path > input1.tmp 				# write all columns but the first
 awk '!/;/ {print $0}' input1.tmp > input2.tmp							 #write all rows not containing a <;>
  tr -d ":="  < input2.tmp > input3.tmp								 # delete all <:=>

 gawk -v "PVscale=$4" -v "loadscale=$5" 'NR== 1 {print $0} NR!=1 && $1!="" {print $1, $2, $3, ($4*loadscale*-1), ($5*PVscale)}' input3.tmp > input.dat
# gawk -v "PVscale=$4" -v "loadscale=$5" 'NR== 1 {print $0} NR!=1 && $1!="" {print $1, $2, $3, ($4*loadscale*-1/1000), ($5*PVscale/10000)}' input3.tmp > input.dat		#multiplies PV and Load by their respective scale

#merge variables and inputparameters
paste input.dat variables.dat > $parsed_path

### clean the mass ###
unset T
rm *tmp input.dat variables.dat

