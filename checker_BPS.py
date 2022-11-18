from parameters_BPS import ParametersBPS_SAC
from environment.blocks_BPS import BPS_SAC
from environment.data_z_sample import ZData
import environment.utils as utils
import numpy as np
from numpy import linalg as LA

run_config = {}
run_config['faff_max'] = 300
run_config['pc_max'] = 10
run_config['window_rate'] = 0.7
run_config['max_window_exp'] = 25
run_config['same_spin_hierarchy'] = True
run_config['dyn_shift'] = 0.
run_config['reward_scale'] = 1.


# ---Instantiating some relevant classes---
params = ParametersBPS_SAC(run_config)
params.g = 0.
zd = ZData()

# ---Kill portion of the z-sample data if required---
zd.kill_data(params.z_kill_list)

cft = BPS_SAC(params, zd)

print(f'g = 0')
objective = cft.compute_g0_function()
#print(objective)
for delta_max in range(1, 21):
    deltas = np.arange(start=1, stop=delta_max+1, step=1)
    obtained = cft.compute_test_vector(delta=deltas, chi=cft.chi)
    mse = LA.norm(objective-obtained)
    print(f'Delta max: {delta_max}, squared mean w.r.t. chi points of (true function - expansion): {mse}')


# ---Instantiating some relevant classes---
params = ParametersBPS_SAC(run_config)
params.g = 'inf'
zd = ZData()

# ---Kill portion of the z-sample data if required---
zd.kill_data(params.z_kill_list)

cft = BPS_SAC(params, zd)

print(f'g = inf')

objective = cft.compute_ginf_function()
#print(objective)
#print(objective)
for delta_max in range(2, 22, 2):
    deltas = np.arange(start=2, stop=delta_max+2, step=2)
    obtained = cft.compute_test_vector(delta=deltas, chi=cft.chi)
    #print(obtained)
    mse = LA.norm(objective-obtained)
    print(f'Delta max: {delta_max}, squared mean w.r.t. chi points of (true function - expansion): {mse}')