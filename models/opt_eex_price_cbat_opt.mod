##################################################################################
##################################################################################
###                                                                            ###
###                              bat.mod                                       ###
###              MODEL OF PV-BATTERY SYSTEM IN FRAUNHOFER ISE                  ###
###                                                                            ###
###     date: 13.9.2012                                                        ###
###     author: Bernhard Wille-Haussmann                                       ###
###                                                                            ###
###     UNITS:                                                                 ###
###     power unit is kW, energy unit is kWh                                   ###
###     money unit is euro                                                     ###
###     time unit: hour                                                        ###
###                                                                            ###
##################################################################################
##################################################################################
#   Day_Ahead_Market adaptation: 23.05.2018 Mostafa Abdelrashied			   ###
#	OPTIMIZATION MODEL for eex elecitricy prices						       ###
##################################################################################

##################################################################################
###       PARAMETERS AND VARIABLES                                             ###
##################################################################################

################################
# Time intervals and parameters#
################################
  	param T > 0 integer;							# Number of values in the data file
	param YYYYMMDD {t in 1..T} > 0 integer;			# String year, month, day
	param hhmmss {t in 1..T} > 0 integer;			# String hour, minute, second
  	param hoy {t in 1..T};							# Hour of year  # linuxtimestamp
  	param step >= 0; 							    # Length of timestep in hours
	param overlap >= 0; 							# Overlap of szenarios in h, must me calculated for real DB

###########
# BATTERY #
###########
   # CONSTANTS
	param sto_soc_start >= 0;						# State of Charge (SoC) at t=0 (0=0%, 1=100%)
 	param sto_cap_max >= 0;							# Storage nominal capacity. Total capacity in kWh
 	param sto_min_charge >= 0;						# Minimum charge power. Battery can not take less than this value. In kW
  	param sto_max_charge >=0;						# Maximum charge power. Battery can not take more that this value. In kW
  	param sto_min_discharge >= 0;					# Minimum discharge power. Battery can not supply less than this value. In kW
  	param sto_max_discharge >=0;					# Maximum discharge power. Battery can not supply more than this value. In kW
	param sto_max_SoC>=0, <=1;						# Maximum SoC to avoid battery overcharging (0=0%, 1=100%)
	param sto_min_SoC>=0, <=1;						# Minimum SoC to avoid battery empty problems (0=0%, 1=100%)
	param sto_loss_charge >= 0;						# Storage losses when charging
  	param sto_loss_discharge >= 0;					# Storage losses when discharging
  	param sto_loss >= 0;							# Storage losses over time per hour (self-discharge and static losses)

   # VARIABLES

  #var sto_soc_c {0..T} >=0, <= 1;					# Storage SOC over time (0=0%, 1=100%)

  #var sto_in_c {1..T} >= 0;						# Energy added to the storage. In kWh.
  #var sto_out_c {1..T} >= 0;						# Energy extracted from the storage. In kWh.

  #var sto_charge_c {1..T} binary;	 				# Boolean flag telling storage is charging.
  #var sto_discharge_c {1..T} binary; 				# Boolean flag telling storage is discharging.

  #var bat_in_c {1..T} >= 0;                 		# Power feed into the storage minus losses. In kW.
  #var bat_out_c {1..T} >= 0;                		# Power taked out from the storage minus losses. In kW.
  var sto_cap_c {0..T};							# Storage capacity over time in kWh.
  var bat_change_c {0..T};						# Power buffer change (output positive). Difference between storage input and output.
  var profit_c {0..T};							# Power buffer change (output positive). Difference between storage input and output.

#########################
# ORIGINAL VARIABLES #
#########################
	# VARIABLES Correct
  #var sto_soc_o {0..T} >=0, <= 1;					# Storage SOC over time (0=0%, 1=100%)

  #var sto_in_o {1..T} >= 0;							# Energy added to the storage. In kWh.
  #var sto_out_o {1..T} >= 0;						# Energy extracted from the storage. In kWh.

  #var sto_charge_o {1..T} binary;	 				# Boolean flag telling storage is charging.
  #var sto_discharge_o {1..T} binary; 				# Boolean flag telling storage is discharging.

  #var bat_in_o {1..T} >= 0;                 		# Power feed into the storage minus losses. In kW.
  #var bat_out_o {1..T} >= 0;                		# Power taked out from the storage minus losses. In kW.
  var sto_cap_o {0..T};							    # Storage capacity over time in kWh.
  var bat_change_o {0..T};							# Power buffer change (output positive). Difference between storage input and output.
  var Day_Ahead_Auction_o {0..T};							# Power buffer change (output positive). Difference between storage input and output.
  var bat_in_o {0..T};							# Power buffer change (output positive). Difference between storage input and output.
  var bat_out_o {0..T};							# Power buffer change (output positive). Difference between storage input and output.
  var cost_o {1..T} >= 0;							# Cost in Euro
  var revenue_o {1..T} >= 0;						# revenue in Euro
  var profit_o {0..T};							# Power buffer change (output positive). Difference between storage input and output.


#########################
# FINAL VARIABLES #
#########################
	# VARIABLES Correct
  var sto_soc {0..T} >=0, <= 1;					# Storage SOC over time (0=0%, 1=100%)
  var sto_cap {0..T};							# Storage capacity over time in kWh.

  var sto_in {1..T} >= 0;						# Energy added to the storage. In kWh.
  var sto_out {1..T} >= 0;						# Energy extracted from the storage. In kWh.

  var sto_charge {1..T} binary;	 				# Boolean flag telling storage is charging.
  var sto_discharge {1..T} binary; 				# Boolean flag telling storage is discharging.

  var bat_in {1..T} >= 0;                 		# Power feed into the storage minus losses. In kW.
  var bat_out {1..T} >= 0;                		# Power taked out from the storage minus losses. In kW.
  var bat_change {0..T};						# Power buffer change (output positive). Difference between storage input and output.
  
  var cost {0..T} >= 0;							# Cost in Euro
  var revenue {1..T} >= 0;						# revenue in Euro
  var marginal_cycle_limit {0..T} >= 0;						# revenue in Euro

  var profit {1..T};								# profit in Euro
  var fullcycle {0..T} >= 0;						# Number of equvalent full-cycles

##########
# DEMAND #
##########

   # CONSTANTS
	param scale_load >= 0;  						# Scale factor for load profile if the load wants to be scaled (

######
# PV #
######

   # CONSTANTS
  	param scale_pv >= 0;  							# Scale factor for PV-profile

######################
# ELECTRICITY MARKET #
######################

   # CONSTANTS
	param Day_Ahead_Auction {1..T};										# Day_ahead_price for electricity in Euro/MWh
	param Intraday_Continuous_Average_Price {1..T};						# Intraday_Continuous_Average_Price for electricity in Euro/MWh
	param Intraday_Auction_15_minute_call {1..T};						# Intraday_Auction_15_minute_call for electricity in Euro/MWh
	param Intraday_Continuous_15_minutes_Average_Price {1..T};			# Intraday_Continuous_15_minutes_Average_Price for electricity in Euro/MWh
	param Intraday_Continuous_30_minutes_Average_Price {1..T};			# Intraday_Continuous_15_minutes_Average_Price for electricity in Euro/MWh




##################
### OPTIMIZING ###
##################

maximize ampl_profit:
	(sum {i in 1..T}  revenue[i]) - (sum {j in 1..T}  cost[j])  - 0 * (sum {k in 1..T}  fullcycle[k]);

#######################
# STORAGE CONSTRAINTS #
#######################
# INTRADAY CORRECTURE CONSTRAINTS
	subject to full_cycles_value {t in 1..T}:   	fullcycle[t]  =  (bat_in[t] * step * (1 - sto_loss_charge)  + bat_out[t] * step * (1 - sto_loss_discharge)) / (2 * sto_cap_max);

	subject to total_cost {t in 1..T}:				cost[t] = Intraday_Auction_15_minute_call[t] * bat_in[t] * step * (1 - sto_loss_charge) / 1000;
	subject to total_revenue {t in 1..T}:			revenue[t] = Intraday_Auction_15_minute_call[t] * bat_out[t] * step * (1 - sto_loss_discharge) / 1000;



	#subject to total_cost_o {t in 1..T}:				cost_o[t] = Day_Ahead_Auction_o[t] * bat_in_o[t] * step * (1 - sto_loss_charge) / 1000;
	#subject to total_revenue_o {t in 1..T}:				revenue_o[t] = Day_Ahead_Auction_o[t] * bat_out_o[t] * step * (1 - sto_loss_discharge) / 1000;

	#subject to profit_calc_o {t in 1..T}:				profit_o[t]  = (revenue_o[t] - cost_o[t]);
	#subject to profit_calc_c {t in 1..T}:				profit_c[t]  = (revenue[t] - cost[t]);

	#subject to profit_calc {t in 1..T}:					profit[t]  = profit_c[t] + profit_o[t];
	
	subject to combined_storage9 {t in 1..T} :   		bat_change_c[t] +  bat_change_o[t] = bat_change[t];
	subject to combined_storage10 {t in 1..T} : 		sto_cap_c[t] + sto_cap_o[t] = sto_cap[t]; 			# calculate storage capacity
	
	
	#subject to th_charge_limit1 {t in 1..T} : 	    	sto_min_charge * sto_charge_c[t] <= bat_in_c[t];
	subject to th_charge_limit11 {t in 1..T} : 	    	sto_min_charge * sto_charge[t] <= bat_in[t];
	
	#subject to th_charge_limit2 {t in 1..T} : 	    	bat_in_c[t] <= sto_charge_c[t] * sto_max_charge ; 
	subject to th_charge_limit22 {t in 1..T} : 	    	bat_in[t] <= sto_charge[t] * sto_max_charge ; 
	
	#subject to th_discharge_limit1 {t in 1..T} :   	sto_min_discharge * sto_discharge_c[t] <= bat_out_c[t];
	subject to th_discharge_limit11 {t in 1..T} :   	sto_min_discharge * sto_discharge[t] <= bat_out[t];
	
	#subject to th_discharge_limit2 {t in 1..T} : 	    bat_out_c[t] <= sto_discharge_c[t] * sto_max_discharge ; 
	subject to th_discharge_limit22 {t in 1..T} : 	    bat_out[t] <= sto_discharge[t] * sto_max_discharge ; 

# calcultion charge and discharge values for STORAGE 
	

	#subject to th1 {t in 1..T}: 				sto_charge_c[t] + sto_discharge_c[t]  <= 1;			               # do not charge and discharge at the same time
	subject to th11 {t in 1..T}: 				sto_charge[t] + sto_discharge[t]  <= 1;			               # do not charge and discharge at the same time
	
	#subject to th2 : 					        sto_soc_c[0] = sto_soc_start;				                       # define storage SOC at t=0
	subject to th22 : 					        sto_soc[0] = sto_soc_start;				                       # define storage SOC at t=0
	
	#subject to th3 {t in 1..T} : 			    bat_change_c[t] = bat_out_c[t] - bat_in_c[t];   	# separating energy flow in input and output
	subject to th33 {t in 1..T} : 			    bat_change[t] = bat_out[t] - bat_in[t];   	# separating energy flow in input and output
	

# to equal HeiPhoSS program
	#subject to th4 {t in 1..T} :				sto_soc_c[t] = sto_soc_c[(t-1)] * (1 - (sto_loss * step)) - (sto_out_c[t] - sto_in_c[t]) / sto_cap_max;   
	subject to th44 {t in 1..T} :				sto_soc[t] = sto_soc[(t-1)] * (1 - (sto_loss * step)) - (sto_out[t] - sto_in[t]) / sto_cap_max;   
												            
	#subject to th5 {t in 1..T} : 				sto_cap_c[t] = sto_soc_c[t] * sto_cap_max; 			# calculate storage capacity
	subject to th55 {t in 1..T} : 				sto_cap[t] = sto_soc[t] * sto_cap_max; 			# calculate storage capacity


	#subject to bat_change_1 {t in 1..T} :		sto_in_c[t]  = bat_in_c[t] * step * (1 - sto_loss_charge);	# bat_in equals storage_in minus losses
	subject to bat_change_11 {t in 1..T} :		sto_in[t]  = bat_in[t] * step * (1 - sto_loss_charge);	# bat_in equals storage_in minus losses
	
	#subject to bat_change_2 {t in 1..T} :		sto_out_c[t] = bat_out_c[t] * step / (1 - sto_loss_discharge);		# bat_out equals storage_out plus losses
	subject to bat_change_22 {t in 1..T} :		sto_out[t] = bat_out[t] * step / (1 - sto_loss_discharge);		# bat_out equals storage_out plus losses
end;