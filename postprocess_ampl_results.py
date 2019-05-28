'''
Created on 07.06.2018

@author: mabdelra
'''
import matplotlib.pyplot as plt

from plot_framework import readamplres

def plot_overview(ts):
    
    plt.close()
    fig, ax1 = plt.subplots(1, figsize=(8,6))
    
    ax1.step(ts.index, ts['Day_Ahead_Auction'], color='blue')
    ax2 = ax1.twinx()
    
    ax2.step(ts.index, ts['sto_soc'], color = 'orange')

    
    ax1.tick_params(axis = 'ts', labelrotation = 45)
    
    plt.show()

    
    


if __name__ == '__main__':
    
    filename = "output/res_0_1000_7_1_0.2_opt_eex_price_bat_opt.mod_AMPL_2017_Referenzdatensatz_1h.dat_1.0_7.0_D_subFiles.dat"    
    ts = readamplres(filename, unix_column='hoy')
    x = ts.sto_in.sum()
    
    print(x)
    exit()

 
     
     
    filename = "output/res_0_1000_7_1_0.5_opt_eex_price_bat_opt.mod_AMPL_2017_Referenzdatensatz_1h.dat_1.0_7.0_D_subFiles.dat"    
    ts = readamplres(filename, unix_column='hoy')
    gain = ts['Profit'].sum()
    print (gain)
    plot_overview(ts)
      
    filename = "output/res_0_1000_7_1_1_opt_eex_price_bat_opt.mod_AMPL_2017_Referenzdatensatz_1h.dat_1.0_7.0_D_subFiles.dat"  
    ts = readamplres(filename, unix_column='hoy')
    gain = ts['Profit'].sum()
    print (gain)
    plot_overview(ts)
    
    