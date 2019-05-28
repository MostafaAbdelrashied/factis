from create_ampl_input import read_spec

# Global Variables
input_folder = "input"
output_folder = "output"
model_folder = "models"
parsed_files_dir = "{}/parsed".format(output_folder)
generic_folder = "{}/generic".format(output_folder)
plot_folder = "{}/Plots".format(generic_folder)
merged_files_dir = "{}/merged".format(output_folder)
bat_spec_dir = "{}/spec.txt".format(output_folder)
bat_dat_filename = "bat.dat"
period_dat_filename = "period.dat"
AMPL_run_file = "simple.run"
AMPL_run_corrected_file = "simple_correct.run"
opt_server_path = "H:/amplopt"

fix_file = 'correct/fix_01.dat'

# Powershell Path
windows_powershell_path = 'C:/WINDOWS/system32/WindowsPowerShell/v1.0/powershell.exe'

# Copying input.dat files from local computer to sx401
input_data_path_local = '{}/{}'.format(input_folder, read_spec()['file_name'])
input_data_path_server = '{}/{}'.format(opt_server_path, input_folder)
input_real_data_path_local = '{}/{}'.format(input_folder, read_spec()['input_real_dat'])
input_real_data_path_server = '{}/{}'.format(opt_server_path, input_folder)
fix_file_path_server = '{}/{}'.format(opt_server_path, fix_file)

# Copying simple.run, bat.dat, period.dat and model.mod files from local computer to sx401
simple_run_path = AMPL_run_file
simple_run_path_corr = AMPL_run_corrected_file
bat_dat_path = bat_dat_filename
per_dat_path = period_dat_filename
model_filename = '{}/{}'.format(model_folder, read_spec()['model'])
model_filename_server = '{}/{}'.format(opt_server_path, model_folder)

# Copying  generic_output.dat and log files from sx401 to local computer

out_dat_path_server = '{}/{}/generic_out.dat'.format(opt_server_path, generic_folder)
out_dat_path_local = '{}/generic_out.dat'.format(generic_folder)

log_dat_path_server = '{}/log_file_correct.txt'.format(opt_server_path)
log_dat_path_server_corr = '{}/log_file_correct.txt'.format(opt_server_path)
log_dat_path_local = 'log_file_correct.txt'
log_dat_path_local_corr = 'log_file_correct.txt'
profit_log = "Profit_log_for_{}.dat".format(read_spec()['file_name'])


def iipath(name):
    """ Give me the name of the individual input file based on the name."""
    standard_iipath = '{0}_subFiles'.format(name)
    return str(standard_iipath)


def oopath(name):
    """Give me the name of the individual OUTPUT file based on the name."""
    standard_oopath = '{0}_subFiles'.format(name)
    return str(standard_oopath)
