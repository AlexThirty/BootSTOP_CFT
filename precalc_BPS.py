from environment.blocks_BPS import BPS_SAC
from parameters_BPS import ParametersBPS_SAC
from environment.data_z_sample import ZData
from environment import utils
import numpy as np

run_config = {}
run_config['faff_max'] = 10000
run_config['pc_max'] = 5
run_config['window_rate'] = 0.5
run_config['max_window_exp'] = 20
run_config['same_spin_hierarchy'] = False
run_config['dyn_shift'] = 0.
run_config['reward_scale'] = 0.001

gs = np.concatenate((np.arange(start=0.01, stop=0.25, step=0.01), np.arange(start=0.25, stop=4.05, step=0.05)))
integral_mode = 0
print(gs)

zd = ZData()


for delta in range(1, 11, 1):
    tmp_name = 'BPS_precalc/blocks_delta' + str(delta) + '.csv'
    for g in gs:
        params = ParametersBPS_SAC(config=run_config, g=g, integral_mode=integral_mode)
        cft = BPS_SAC(params, zd)
        value = cft.precalc_block(params.shifts_deltas)
        utils.output_to_file(file_name=tmp_name, output=value)
        
for delta in range(1, 11, 1):
    tmp_name = 'BPS_precalc/integral1_delta' + str(delta) + '.csv'
    for g in gs:
        params = ParametersBPS_SAC(config=run_config, g=g, integral_mode=integral_mode)
        cft = BPS_SAC(params, zd)
        value = cft.integral_1(delta=delta)
        utils.output_to_file(file_name=tmp_name, output=value)
        
for delta in range(1, 11, 1):
    tmp_name = 'BPS_precalc/integral2_delta' + str(delta) + '.csv'
    for g in gs:
        params = ParametersBPS_SAC(config=run_config, g=g, integral_mode=integral_mode)
        cft = BPS_SAC(params, zd)
        value = cft.integral_2(delta=delta)
        utils.output_to_file(file_name=tmp_name, output=value)