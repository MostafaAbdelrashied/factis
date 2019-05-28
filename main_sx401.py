#!/usr/bin/python
mabdelrashied 05.05.2018

from ampl import loopSubFiles_sx401


# FUNCTIONS
###
def write_bat_dat(step: float, bat_scale: int, pv_scale: int, load_scale: int, Cvalue: int, charge_loss: int,
                  discharge_loss: float, selfdischarge_loss: int) -> dict:
    """
    Funtion to write bat.dat file with the parameters required for the optimization.
    :return: paramDict: Dictionary contains all battery specification
    :param step: The time resolution (in hours) for the market that the battery is trading inside e.g. 1 for Day-Ahead, 0.25 for Intraday Auction
    :param bat_scale: The nominal capacity of the battery in kWh
    :param pv_scale: Scale factor for PV-profile ---> PV1_Pel * scale_pv = PV Power generated
    :param load_scale: The load capacity
    :param Cvalue: C-rate of battery
    :param charge_loss: the losses incurred during charing the battery
    :param discharge_loss: the losses incurred during discharging the battery
    :param selfdischarge_loss: internal chemical reactions reduce the stored charge of the battery without any connection between the electrodes.
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


def change_model(model, file_path="simple.run"):
    """
    Change the name of the model in simple.run file
    :param model: AMPL's model that has to be changed
    :param file_path: write the model name in the simple.run file
    """
    lines = open(file_path, 'r').readlines()
    lines[3] = "model models/{};\n".format(model)
    with open(file_path, 'w') as runfile:
        for line in lines:
            runfile.write(line)


def get_line_no(file_path, str):
    """
    Get the line number of a certain string
    :param file_path: The model path
    :param str: the string needed to be searched for
    :return: num: number of line that contains the string under modification
    """
    with open(file_path) as myFile:
        for num, line in enumerate(myFile, 1):
            if str in line:
                return num


def change_price_col(price_col, model):
    """
    Change the name of the price column in model.mod file
    :param price_col: string of the priice needed to be change to
    :param model: AMPL's model name
    """
    file_path = 'models/{}'.format(model)
    line_no = get_line_no(file_path, str='total_cost')
    line_no2 = get_line_no(file_path, str='total_revenue')
    with open(file_path, 'r+') as f:
        lines = f.readlines()
        lines[
            line_no - 1] = '\tsubject to total_cost {t in 1..T}:\t\t\t\tcost[t] = %s[t] * bat_in[t] * step * (1 - sto_loss_charge) / 1000;\n' % (
            price_col)
        lines[
            line_no2 - 1] = '\tsubject to total_revenue {t in 1..T}:\t\t\trevenue[t] = %s[t] * bat_out[t] * step * (1 - sto_loss_discharge) / 1000;\n' % (
            price_col)
    with open(file_path, 'w') as f2:
        for line in lines:
            f2.write(line)


def profit_per_cycle_limit(model, profit):
    """
    Overwriting the profit limit value in the AMPL model to a certain value
    :param model:  model name
    :param profit: the profit limit value that is needed to be overwritten in AMPL's model
    """
    file_path = 'models/{}'.format(model)
    line_no = get_line_no(file_path, str='maximize ampl_profit')
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines[
            line_no] = '\t(sum {i in 1..T}  revenue[i]) - (sum {j in 1..T}  cost[j])  - %s * (sum {k in 1..T}  fullcycle[k]);\n' % (
            profit)
    with open(file_path, 'w') as f2:
        for line in lines:
            f2.write(line)


def sim_params(filename):
    """

    :param filename: filename that contains price_col
    :return: price_cols, discharges, crate, step: all result file specification
    """
    time_res = filename.split('_')[5].replace('.dat', '')
    if time_res == '1h':
        price_cols = ['Day_Ahead_Auction']
        dischargerates = [0.5]
        crate = 0.5
        step = 1
    elif time_res == '15min':
        price_cols = ['Intraday_Continuous_15_minutes_Average_Price']
        dischargerates = [0.2]
        crate = 0.2
        step = 0.25

    return price_cols, dischargerates, crate, step


def eff_values(rate, crate):
    """
    Check if the c-rate is the required one, perform additional efficiency anaylsis on it
    :param rate: rate at which results is optimized with
    :param crate: the c-rate where different efficienies should be optimized with
    :return:
    """
    if rate == crate:
        roundtripeff = [0.9]
        # 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1
    else:
        roundtripeff = [0.9]
    return roundtripeff


# CORE PROGRAM


if __name__ == "__main__":

    #################################################################################################################
    # DEFINITION OF CONSTANTS
    #################################################################################################################

    prediction_vs_real = False  # True: optimization will be calculated using prediction and results will be used
    # for the real-time simulation
    # False: optimization will be calculated using input data without doing any process afterwards.
    pv_load_samefile = True  # True: means pv and load prediction data are in the same file
    # False: means pv and load are in separate files. Variable "hoy" must be contained in only one of them
    loopStart = 0  # starting file number. Use "0" to start with first file
    loopEnd = 1  # ending file number. Use "0" to end with the last file

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

    epex_prices_inputfile_list = ['energycharts_downloads_price_2018_en_15min.dat_24.0_12.0_H_subFiles']  # ,
    # energycharts_downloads_price_2017_en_1h.dat_365.0_0.0_D_subFiles
    # energycharts_downloads_price_2017_en_15min.dat_365.0_0.0_D_subFiles
    # energycharts_downloads_price_2017_en_1h.dat_24.0_12.0_H_subFiles
    # energycharts_downloads_price_2017_en_15min.dat_24.0_12.0_H_subFiles

    load_inputfile_list = [
        "loadChriWi_shifted1day.dat_3.0_14.75_H_subFiles"]  # , "loadSMA_shifted1day.dat_3.0_14.75_H_subFiles"]
    # ['2012_LoadChriWi_1min_real.dat_1.0_0.0_D_subFiles', '2012_LoadChriWi_15min_real.dat_1.0_0.0_D_subFiles', '2012_LoadChriWi_1sec_real.dat_6.0_0.0_H_subFiles']

    pv_load_realdatafile_list = [
        "res_0_1000_0_0_1_1_energycharts_downloads_price_2017_en_1h.dat_24.0_12.0_H_subFiles_Day_Ahead_Auction.dat"]
    # ["2012_PV_LoadChriWi_real.dat", "2012_PV_LoadSMA_real.dat"]

    # number_of_scenarios = 1      # number of files in every list. Number of scenarios to be calculated
    # price_cols = [['Intraday_Auction_15_minute_call', 'Intraday_Continuous_15_minutes_Average_Price'], ['Day_Ahead_Auction', 'Intraday_Continuous_Average_Price']]  # ['Intraday_Auction_15_minute_call', 'Intraday_Continuous_15_minutes_Average_Price']
    # Day_Ahead_Auction, Intraday_Continuous_Average_Price, Intraday_Auction_15_minute_call, Intraday_Continuous_15_minutes_Average_Price
    #################################################################################################################
    # LISTS OF CASES.             Variables to test
    #################################################################################################################

    bat_scales = [
        1000]  # [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] ---> Storage nominal capacity. Total capacity in kWh
    pv_scales = [
        0]  # [2.5, 5.0, 10.0]#, 2.5, 10.0] ---> Scale factor for PV-profile ---> PV1_Pel * scale_pv = PV Power generated
    load_scales = [
        0]  # [2.0, 4.0, 6.0]---> Scale factor for load profile if the load wants to be scaled ---> Load_Pel * scale_load = energy consumed by the load
    # dischargerate = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4],
    #                 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]]
    # 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4]
    # ---> A C-rate is a measure of the rate at which a battery is discharged relative to its maximum capacity
    profit_limits = [0]
    # 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10
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
    simplerun = 'simple.run'

    ###################################################################################################################
    # PROGRAM EXECUTION
    ###################################################################################################################

    for model in modellist:
        change_model(model, simplerun)
        for i in range(0, len(epex_prices_inputfile_list)):
            price_cols, dischargerates, crate, step = sim_params(epex_prices_inputfile_list[i])
            for price_col in price_cols:
                change_price_col(price_col, model)
                for profit_limit in profit_limits:
                    profit_per_cycle_limit(model, profit_limit)
                    for rate in dischargerates:
                        roundtripeff = eff_values(rate, crate)
                        for bat_scale in bat_scales:
                            for pv_scale in pv_scales:
                                for load_scale in load_scales:
                                    for eff in roundtripeff:
                                        charge_loss = 1 - eff ** (
                                                1 / 2)  # 0.0494, ## 98%BatteryEff * 97%InverterEff = 95.06%Eff = 0.0494%Losses ## Losses used in HeiPhoss=0.049210854
                                        discharge_loss = 1 - eff ** (
                                                1 / 2)  # 0.0494, ## 98%BatteryEff * 97%InverterEff = 95.06%Eff = 0.0494%Losses ## Losses used in HeiPhoss=0.049210854
                                        selfdischarge_loss = 0  # 1-(1-0.07)**(step[i]/(30*24))
                                        szenname = "{}_{}_{}_{}_{}_{}_{}".format(i, bat_scale, pv_scale, load_scale,
                                                                                 rate,
                                                                                 eff, profit_limit,
                                                                                 model)
                                        paramDict = write_bat_dat(step, bat_scale, pv_scale, load_scale, rate,
                                                                  charge_loss,
                                                                  discharge_loss, selfdischarge_loss)
                                        paramDict['model'] = model
                                        if prediction_vs_real is False:
                                            if pv_load_samefile is True:
                                                paramDict['input_dat_1'] = epex_prices_inputfile_list[i]
                                                paramDict['input_dat_2'] = ''
                                                paramDict['input_real_dat'] = ''
                                                loopSubFiles_sx401(epex_prices_inputfile_list[i], loopStart, loopEnd,
                                                                   prediction_vs_real,
                                                                   szenname, paramDict, price_col=price_col,
                                                                   simple_run=simplerun, copy_input=False)
                                            else:
                                                paramDict['input_dat_1'] = epex_prices_inputfile_list[i]
                                                paramDict['input_dat_2'] = load_inputfile_list[i]
                                                paramDict['input_real_dat'] = ''
                                                loopSubFiles_sx401(epex_prices_inputfile_list[i], loopStart, loopEnd,
                                                                   prediction_vs_real,
                                                                   szenname, paramDict, load_inputfile_list[i],
                                                                   price_col=price_col,
                                                                   simple_run=simplerun, copy_input=False)
                                        else:
                                            if pv_load_samefile is True:
                                                paramDict['input_dat_1'] = epex_prices_inputfile_list[i]
                                                paramDict['input_dat_2'] = ''
                                                paramDict['input_real_dat'] = pv_load_realdatafile_list[i]
                                                loopSubFiles_sx401(epex_prices_inputfile_list[i], loopStart, loopEnd,
                                                                   prediction_vs_real,
                                                                   szenname, paramDict, pv_load_realdatafile_list[i],
                                                                   price_col=price_col,
                                                                   simple_run=simplerun, copy_input=False)
                                            else:
                                                paramDict['input_dat_1'] = epex_prices_inputfile_list[i]
                                                paramDict['input_dat_2'] = load_inputfile_list[i]
                                                paramDict['input_real_dat'] = pv_load_realdatafile_list[i]
                                                loopSubFiles_sx401(epex_prices_inputfile_list[i], loopStart, loopEnd,
                                                                   prediction_vs_real,
                                                                   szenname, paramDict, pv_load_realdatafile_list[i],
                                                                   load_inputfile_list[i], price_col=price_col,
                                                                   simple_run=simplerun, copy_input=False)
