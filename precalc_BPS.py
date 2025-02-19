from environment.blocks_BPS import BPS_SAC
from parameters_BPS import ParametersBPS_SAC
from environment.data_z_sample import ZData
from environment import utils
import numpy as np
import values_BPS

deltas = []
deltas.append(values_BPS.delta1)
deltas.append(values_BPS.delta2)
deltas.append(values_BPS.delta3)
deltas.append(values_BPS.delta4)
deltas.append(values_BPS.delta5)
deltas.append(values_BPS.delta6)
deltas.append(values_BPS.delta7)
deltas.append(values_BPS.delta8)
deltas.append(values_BPS.delta9)
deltas.append(values_BPS.delta10)

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


for delta in range(1, 11, 1):
    tmp_name = 'BPS_precalc/blocks_delta' + str(delta) + '.csv'
    for g in gs:
        delta_val = deltas[delta-1][str(g)]
        params = ParametersBPS_SAC(config=run_config, g=g, integral_mode=integral_mode)
        cft = BPS_SAC(params, zd, [], [], [])
        value = cft.precalc_block(delta=delta_val).real
        utils.output_to_file(file_name=tmp_name, output=value)
        
for delta in range(1, 11, 1):
    tmp_name = 'BPS_precalc/integral1_delta' + str(delta) + '.csv'
    for g in gs:
        delta_val = deltas[delta-1][str(g)]
        params = ParametersBPS_SAC(config=run_config, g=g, integral_mode=integral_mode)
        cft = BPS_SAC(params, zd, [], [], [])
        value = cft.integral_1(delta=delta_val)
        utils.output_to_file(file_name=tmp_name, output=[value])
        
for delta in range(1, 11, 1):
    tmp_name = 'BPS_precalc/integral2_delta' + str(delta) + '.csv'
    for g in gs:
        delta_val = deltas[delta-1][str(g)]
        params = ParametersBPS_SAC(config=run_config, g=g, integral_mode=integral_mode)
        cft = BPS_SAC(params, zd, [], [], [])
        value = cft.integral_2(delta=delta_val)
        utils.output_to_file(file_name=tmp_name, output=[value])