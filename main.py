#!/usr/bin/python

from __future__ import print_function

from ampl import loopSubFiles
from ampl import loopSubFiles_sx401
from subprocess import call

### FUNCTIONS
###

def writebatdat(step, bat_scale, pv_scale, load_scale, Cvalue, charge_loss, discharge_loss, selfdischarge_loss):
    """
    Funtion to write bat.dat file with the parameters required for the optimization.
    """
    
    paramDict = { 
            'step' : step,
            'sto_cap_max' : bat_scale, 
            'sto_min_charge' : 0, 
            'sto_max_charge' : bat_scale*Cvalue, 
            'sto_min_discharge' : 0, 
            'sto_max_discharge' : bat_scale, 
            'sto_loss_charge' : charge_loss,
            'sto_loss_discharge' : discharge_loss, 
            'sto_loss' :  selfdischarge_loss,
            'sto_max_SoC' : 1, 
            'sto_min_SoC' : 0, 
            'scale_load' : load_scale, 
            'scale_pv' : pv_scale,
            'feed_in_limit' : 0.6
             }
    template = "######################\n# PARAMETER FOR AMPL build by main.py #\n######################\n"    
    
    for key in paramDict.keys():
        template += 'param {0:s} := {1:.10f};\n'.format(key, paramDict[key])
        
    with open('bat.dat', 'w') as parameterFile:
        parameterFile.write( template )
        
    return paramDict    


def changemodel(model):
    """
    Change the old model with a new predefined one
    """
    file_path = "simple.run"
    lines = open(file_path, 'r').readlines()
    lines[3] =  "model models/{};\n".format(model)
    with open(file_path, 'w') as runfile:
        for line in lines: 
            runfile.write(line)



### CORE PROGRAM

if __name__ == "__main__":

    #################################################################################################################
    # DEFINITION OF CONSTANTS
    #################################################################################################################

    step                   =[0.25]
                            # [0.0166666667, 0.25, 0.000277778]# = 1min, 15min, 1sec    

    charge_loss = 0.049210854            #0.0494, ## 98%BatteryEff * 97%InverterEff = 95.06%Eff = 0.0494%Losses ## Losses used in HeiPhoss=0.049210854
    discharge_loss = 0.049210854         #0.0494, ## 98%BatteryEff * 97%InverterEff = 95.06%Eff = 0.0494%Losses ## Losses used in HeiPhoss=0.049210854
    
    prediction_vs_real = 0      # 1: optimization will be calculated using prediction and results will be used for the real-time simulation
                                # 0: optimization will be calculated using input data without doing any process afterwards.
    
    pv_load_samefile = 1        # 1: means pv and load prediction data are in the same file
                                # 0: means pv and load are in separate files. Variable "hoy" must be contained in only one of them 
    
    loopstart = 0                  # starting file number. Use "0" to start with first file
    loopend = 3                 # ending file number. Use None to end with the last file
    
    
    ############################################################################################################################
    # LIST OF FILES.              All the lists must contain the same number of files to be read. 
    #                             If a same file is used for several scenarios, it can be written several times
    # pv_load_inputfile_list:     List of files with input data. Can contain only PV data or PV and LOAD data. 
    # load_inputfile_list:        If pv_load_samefile = 0 this file will be read to get the load values, 
    #                             otherwise it will be ignored assuming the load data is contained in the file read from pv_load_inputfile_list
    # pv_load_realdatafile_list:  If prediction_vs_real = 1 this file will be read as the real values to handle the real time operation 
    #                             using the results of the optimization
    #
    #                             If load_inputfile_list and/or pv_load_realdatafile_list are used, they must contain the same number of files 
    #                             as the contained in pv_load_inputfile_list. The lists will be matched by using the files in the same position from 
    #                             all lists, therefore all lists must contain the same number of files to be read (if they are used)       
    #                             If a same file is used for several scenarios, it can be written several times
    ############################################################################################################################

    pv_load_inputfile_list = ['AMPL_2012_Referenzdatensatz_15min.dat_7.0_15.0_D_subFiles']#, '2012_pv_prediction.dat_3.0_14.75_H_subFiles'] ['AMPL_2012_SMA_Referenzdatensatz_1min.dat_45.0_15.0_D_subFiles']
                            # ['2013_pv_1min.dat_1.0_0.0_D_subFiles', '2013_pv_15min.dat_1.0_0.0_D_subFiles', '2013_pv_1sec.dat_6.0_0.0_H_subFiles']
                           
    load_inputfile_list    = ["loadChriWi_shifted1day.dat_3.0_14.75_H_subFiles"]#, "loadSMA_shifted1day.dat_3.0_14.75_H_subFiles"]
                            # ['2012_LoadChriWi_1min_real.dat_1.0_0.0_D_subFiles', '2012_LoadChriWi_15min_real.dat_1.0_0.0_D_subFiles', '2012_LoadChriWi_1sec_real.dat_6.0_0.0_H_subFiles']
                           
    pv_load_realdatafile_list = ["2012_PV_LoadChriWi_real.dat"]
                                #["2012_PV_LoadChriWi_real.dat", "2012_PV_LoadSMA_real.dat"]
    


    number_of_scenarios = 1      # number of files in every list. Number of scenarios to be calculated


    #################################################################################################################
    # LISTS OF CASES.             Variables to test
    #################################################################################################################

    bat_scales= [10] #[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]#, 3.0, 5.0] 
    pv_scales = [7]#[2.5, 5.0, 10.0]#, 2.5, 10.0] 
    load_scales = [1]#[2.0, 4.0, 6.0] 
    dischargerate = [1]#[0.7, 1.0, 0.5]

    #################################################################################################################
    # LIST OF MODELS
    #    opt_Trad.mod                       # Conventional operation without optimization
    #    opt_WithoutBatDischarge.mod        # Optimization without battery discharge
    #    opt_MinLosses.mod                  # Optimization discharging battery only to avoid cut-off losses by the feed in limit
    #    opt_MinGridFeedIn.mod              # Optimization discharging as much as possible to reduce the peak of power fed into the grid
    #    opt_Dinamic.mod                    # Optimization following a desired SoC level according to the expected energy excess.
    #                                         opt_dinamic.mod only works correctly with prediction_vs_real = 1
    #################################################################################################################    

    modellist = ["opt_Trad.mod"]


    ###################################################################################################################
    # PROGRAM EXECUTION
    ###################################################################################################################

    for model in modellist:
        changemodel(model)
        for i in range(0, len(pv_load_inputfile_list)):
            for rate in dischargerate:                                                     
                for bat_scale in bat_scales:
                    for pv_scale in pv_scales:
                        for load_scale in load_scales:
                            selfdischarge_loss = 1-(1-0.07)**(step[i]/(30*24))
                            szenname = "{}_{}_{}_{}_{}_{}".format(i, bat_scale, pv_scale, load_scale, rate, model)
                            paramDict = writebatdat(step[i], bat_scale, pv_scale, load_scale, rate, charge_loss, discharge_loss, selfdischarge_loss)
                            paramDict['model'] = model
                            if prediction_vs_real==0:
                                if pv_load_samefile==1:
                                    paramDict['inputfile1'] = pv_load_inputfile_list[i]
                                    paramDict['inputfile2'] = ''
                                    paramDict['realdatafile'] = ''
                                    loopSubFiles_sx401(pv_load_inputfile_list[i], loopstart, loopend, prediction_vs_real, szenname, paramDict)
                                    # loopSubFiles(pv_load_inputfile_list[i], loopstart, loopend, prediction_vs_real, szenname, paramDict)
                                    # call(['python', 'output/build_totals.py'])
                                else:
                                    paramDict['inputfile1'] = pv_load_inputfile_list[i]
                                    paramDict['inputfile2'] = load_inputfile_list[i]
                                    paramDict['realdatafile'] = ''
                                    loopSubFiles(pv_load_inputfile_list[i], loopstart, loopend, prediction_vs_real, szenname, paramDict, "", load_inputfile_list[i])
                                    call(['python', 'output/build_totals.py'])
                            else:
                                if pv_load_samefile==1:
                                    paramDict['inputfile1'] = pv_load_inputfile_list[i]
                                    paramDict['inputfile2'] = ''
                                    paramDict['realdatafile'] = pv_load_realdatafile_list[i]
                                    loopSubFiles(pv_load_inputfile_list[i], loopstart, loopend, prediction_vs_real, szenname, paramDict, pv_load_realdatafile_list[i])
                                    call(['python', 'output/build_totals_merged.py'])
                                else:
                                    paramDict['inputfile1'] = pv_load_inputfile_list[i]
                                    paramDict['inputfile2'] = load_inputfile_list[i]
                                    paramDict['realdatafile'] = pv_load_realdatafile_list[i]
                                    loopSubFiles(pv_load_inputfile_list[i], loopstart, loopend, prediction_vs_real, szenname, paramDict, pv_load_realdatafile_list[i], load_inputfile_list[i])
                                    call(['python', 'output/build_totals_merged.py'])
