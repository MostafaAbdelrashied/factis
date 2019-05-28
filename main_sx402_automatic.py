#!/usr/bin/python

from ampl import loopSubFiles_sx401
# FUNCTIONS
###


def writebatdat(step, bat_scale, pv_scale, load_scale, Cvalue, charge_loss, discharge_loss, selfdischarge_loss):
    """
    Funtion to write bat.dat file with the parameters required for the optimization.
    """

    paramDict = {
        'step': step,
        'sto_cap_max': bat_scale,
        'sto_min_charge': 0,
        'sto_max_charge': bat_scale * Cvalue,
        'sto_min_discharge': 0,
        'sto_max_discharge': bat_scale * Cvalue,
        'sto_loss_charge': charge_loss,
        'sto_loss_discharge': discharge_loss,
        'sto_loss': selfdischarge_loss,
        'scale_load': load_scale,
        'scale_pv': pv_scale
    }
    template = "######################\n# PARAMETER FOR AMPL build by main.py #\n######################\n"

    for key, value in paramDict.items():
        template += 'param {0:s} := {1:.10f};\n'.format(key, value)

    with open('bat.dat', 'w') as parameterFile:
        parameterFile.write(template)

    return paramDict


def changemodel(model):
    """
    Change the name of the model in simple.run file
    """

    file_path = "simple.run"
    lines = open(file_path, 'r').readlines()
    lines[3] = "model models/{};\n".format(model)
    with open(file_path, 'w') as runfile:
        for line in lines:
            runfile.write(line)


def change_price_col(price_col):
    """
    Change the name of the price column in model.mod file
    """
    file_path = 'models/opt_eex_price_bat_opt.mod'

    def get_line_no():
        with open(file_path) as myFile:
            for num, line in enumerate(myFile, 1):
                if 'maximize ampl_profit:' in line:
                    return num
    price_line_no = get_line_no()
    lines = open(file_path, 'r').readlines()
    lines[price_line_no] = '\t(sum {i in  1..T}  %s[i]  * step * bat_out[i] / 1000)  - (sum {j in  1..T}  %s[j]  * step * bat_in[j] / 1000 );\n' % (price_col, price_col)
    with open(file_path, 'w') as runfile:
        for line in lines:
            runfile.write(line)


def step_finder(list):
    step = []
    for item in list:
        if '1h' in item:
            step += [1]
        elif '30min' in item:
            step += [0.5]
        elif '15min' in item:
            step += [0.25]
    return step


def sim_params(filename):
    time_res = filename.split('_')[5].replace('.dat', '')
    if time_res == '1h':
        price_cols = ['Day_Ahead_Auction', 'Intraday_Continuous_Average_Price']
        dischargerates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        crate = 0.5
        step = 1
    elif time_res == '15min':
        price_cols = ['Intraday_Auction_15_minute_call', 'Intraday_Continuous_15_minutes_Average_Price']
        dischargerates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4]
        crate = 4
        step = 0.25

    return price_cols, dischargerates, crate, step


def eff_values(rate, crate):
    if rate == crate:
        roundtripeff = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    else:
        roundtripeff = [1]
    return roundtripeff


# CORE PROGRAM


if __name__ == "__main__":

    #################################################################################################################
    # DEFINITION OF CONSTANTS
    #################################################################################################################

    prediction_vs_real = False      # True: optimization will be calculated using prediction and results will be used for the real-time simulation
    # False: optimization will be calculated using input data without doing any process afterwards.
    pv_load_samefile = True         # True: means pv and load prediction data are in the same file
    # False: means pv and load are in separate files. Variable "hoy" must be contained in only one of them
    loopStart = 0      # starting file number. Use "0" to start with first file
    loopEnd = 0      # ending file number. Use "0" to end with the last file

    ############################################################################################################################
    # LIST OF FILES.              All the lists must contain the same number of files to be read.
    #                             If a same file is used for several scenarios, it can be written several times
    # epex_prices_inputfile_list: List of files with input data. Can contain only PV data or PV and LOAD data.
    # load_inputfile_list:        If pv_load_samefile = 0 this file will be read to get the load values,
    #                             otherwise it will be ignored assuming the load data is contained in the file read from epex_prices_inputfile_list
    # pv_load_realdatafile_list:  If prediction_vs_real = 1 this file will be read as the real values to handle the real time operation
    #                             using the results of the optimization
    #
    #                             If load_inputfile_list and/or pv_load_realdatafile_list are used, they must contain the same number of files
    #                             as the contained in epex_prices_inputfile_list. The lists will be matched by using the files in the same position from
    #                             all lists, therefore all lists must contain the same number of files to be read (if they are used)
    #                             If a same file is used for several scenarios, it can be written several times
    ############################################################################################################################

    epex_prices_inputfile_list = ['energycharts_downloads_price_2017_en_15min.dat_24.0_12.0_H_subFiles', 'energycharts_downloads_price_2017_en_1h.dat_24.0_12.0_H_subFiles']  # ,
    #'energycharts_downloads_price_2017_en_15min.dat_24.0_12.0_H_subFiles']
    # , '2012_pv_prediction.dat_3.0_14.75_H_subFiles'] ['AMPL_2012_SMA_Referenzdatensatz_1min.dat_45.0_15.0_D_subFiles']
    #['2013_pv_1min.dat_1.0_0.0_D_subFiles', '2013_pv_15min.dat_1.0_0.0_D_subFiles', '2013_pv_1sec.dat_6.0_0.0_H_subFiles']

    # [0.0166666667, 0.25, 0.000277778]# = 1min, 15min, 1sec

    load_inputfile_list = ["loadChriWi_shifted1day.dat_3.0_14.75_H_subFiles"]  # , "loadSMA_shifted1day.dat_3.0_14.75_H_subFiles"]
    #['2012_LoadChriWi_1min_real.dat_1.0_0.0_D_subFiles', '2012_LoadChriWi_15min_real.dat_1.0_0.0_D_subFiles', '2012_LoadChriWi_1sec_real.dat_6.0_0.0_H_subFiles']

    pv_load_realdatafile_list = ["combined_downloads_price_2017_en_1h.dat"]
    #["2012_PV_LoadChriWi_real.dat", "2012_PV_LoadSMA_real.dat"]

    # number_of_scenarios = 1      # number of files in every list. Number of scenarios to be calculated
    # price_cols = [['Intraday_Auction_15_minute_call', 'Intraday_Continuous_15_minutes_Average_Price'], ['Day_Ahead_Auction', 'Intraday_Continuous_Average_Price']]  # ['Intraday_Auction_15_minute_call', 'Intraday_Continuous_15_minutes_Average_Price']
    # Day_Ahead_Auction, Intraday_Continuous_Average_Price, Intraday_Auction_15_minute_call, Intraday_Continuous_15_minutes_Average_Price
    #################################################################################################################
    # LISTS OF CASES.             Variables to test
    #################################################################################################################

    bat_scales = [1000]  # [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] ---> Storage nominal capacity. Total capacity in kWh
    pv_scales = [0]  # [2.5, 5.0, 10.0]#, 2.5, 10.0] ---> Scale factor for PV-profile ---> PV1_Pel * scale_pv = PV Power generated
    load_scales = [0]  # [2.0, 4.0, 6.0]---> Scale factor for load profile if the load wants to be scaled ---> Load_Pel * scale_load = energy consumed by the load
    # dischargerate = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4],
    #                 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]]
    # 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4]
    # ---> A C-rate is a measure of the rate at which a battery is discharged relative to its maximum capacity

    #################################################################################################################
    # LIST OF MODELS
    #    opt_Trad.mod                       # Conventional operation without optimization
    #    opt_WithoutBatDischarge.mod        # Optimization without battery discharge
    #    opt_MinLosses.mod                  # Optimization discharging battery only to avoid cut-off losses by the feed in limit
    #    opt_MinGridFeedIn.mod              # Optimization discharging as much as possible to reduce the peak of power fed into the grid
    #    opt_Dinamic.mod                    # Optimization following a desired SoC level according to the expected energy excess.
    #                                         opt_dinamic.mod only works correctly with prediction_vs_real = 1
    #################################################################################################################

    modellist = ['opt_eex_price_bat_opt.mod']

    ###################################################################################################################
    # PROGRAM EXECUTION
    ###################################################################################################################

    for model in modellist:
        changemodel(model)
        for i in range(0, len(epex_prices_inputfile_list)):
            price_cols, dischargerates, crate, step = sim_params(epex_prices_inputfile_list[i])
            for price_col in price_cols:
                change_price_col(price_col)
                for rate in dischargerates:
                    roundtripeff = eff_values(rate, crate)
                    for bat_scale in bat_scales:
                        for pv_scale in pv_scales:
                            for load_scale in load_scales:
                                for eff in roundtripeff:
                                    charge_loss = 1 - eff**(1 / 2)   # 0.0494, ## 98%BatteryEff * 97%InverterEff = 95.06%Eff = 0.0494%Losses ## Losses used in HeiPhoss=0.049210854
                                    discharge_loss = 1 - eff**(1 / 2)  # 0.0494, ## 98%BatteryEff * 97%InverterEff = 95.06%Eff = 0.0494%Losses ## Losses used in HeiPhoss=0.049210854
                                    selfdischarge_loss = 0  # 1-(1-0.07)**(step[i]/(30*24))
                                    szenname = "{}_{}_{}_{}_{}_{}".format(i, bat_scale, pv_scale, load_scale, rate, eff, model)
                                    paramDict = writebatdat(step, bat_scale, pv_scale, load_scale, rate, charge_loss, discharge_loss, selfdischarge_loss)
                                    paramDict['model'] = model
                                    if prediction_vs_real is False:
                                        if pv_load_samefile is True:
                                            paramDict['inputfile1'] = epex_prices_inputfile_list[i]
                                            paramDict['inputfile2'] = ''
                                            paramDict['realdatafile'] = ''
                                            loopSubFiles_sx401(epex_prices_inputfile_list[i], loopStart, loopEnd, prediction_vs_real,
                                                               szenname, paramDict, price_col=price_col)
                                        else:
                                            paramDict['inputfile1'] = epex_prices_inputfile_list[i]
                                            paramDict['inputfile2'] = load_inputfile_list[i]
                                            paramDict['realdatafile'] = ''
                                            loopSubFiles_sx401(epex_prices_inputfile_list[i], loopStart, loopEnd, prediction_vs_real,
                                                               szenname, paramDict, load_inputfile_list[i], price_col=price_col)
                                    else:
                                        if pv_load_samefile is True:
                                            paramDict['inputfile1'] = epex_prices_inputfile_list[i]
                                            paramDict['inputfile2'] = ''
                                            paramDict['realdatafile'] = pv_load_realdatafile_list[i]
                                            loopSubFiles_sx401(epex_prices_inputfile_list[i], loopStart, loopEnd, prediction_vs_real,
                                                               szenname, paramDict, pv_load_realdatafile_list[i], price_col=price_col)
                                        else:
                                            paramDict['inputfile1'] = epex_prices_inputfile_list[i]
                                            paramDict['inputfile2'] = load_inputfile_list[i]
                                            paramDict['realdatafile'] = pv_load_realdatafile_list[i]
                                            loopSubFiles_sx401(epex_prices_inputfile_list[i], loopStart, loopEnd, prediction_vs_real,
                                                               szenname, paramDict, pv_load_realdatafile_list[i], load_inputfile_list[i], price_col=price_col)
