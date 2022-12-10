from environment.blocks_BPS import BPS_SAC
from parameters_BPS import ParametersBPS_SAC
from environment.data_z_sample import ZData
from environment import utils
import numpy as np
import values_BPS

run_config = {}
run_config['faff_max'] = 10000
run_config['pc_max'] = 5
run_config['window_rate'] = 0.5
run_config['max_window_exp'] = 20
run_config['same_spin_hierarchy'] = False
run_config['dyn_shift'] = 0.
run_config['reward_scale'] = 0.001

gs = np.concatenate((np.arange(start=0.01, stop=0.25, step=0.01),
                     np.arange(start=0.25, stop=4.05, step=0.05),
                     #np.arange(start=4.25, stop=5.25, step=0.25)
                     ))
integral_mode = 0
print(gs)
gs = np.around(gs, decimals=2)
zd = ZData()
g=1.

params = ParametersBPS_SAC(config=run_config, g=g, integral_mode=integral_mode)
start = params.delta_start
stop = params.delta_end_increment
step = params.delta_sep
cft = BPS_SAC(params, zd, [], [], [])

deltas = np.arange(start=start, stop=stop, step=step)

tmp_name = 'BPS_precalc_free/blocks.csv'
for i in range(len(deltas)):
    delta = deltas[i]
    values = cft.precalc_block(delta=delta).real
    utils.output_to_file(file_name=tmp_name, output=values)

tmp_name = 'BPS_precalc_free/integral1.csv'
for i in range(len(deltas)):
    delta = deltas[i]
    values = cft.integral_1(delta=delta)
    utils.output_to_file(file_name=tmp_name, output=[values])
        
tmp_name = 'BPS_precalc_free/integral2.csv'
for i in range(len(deltas)):
    delta = deltas[i]
    values = cft.integral_2(delta=delta)
    utils.output_to_file(file_name=tmp_name, output=[values])