from parameters_ising2D import ParametersIsing2D_SAC
from environment.blocks_ising2D import Ising2D_SAC
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
params = ParametersIsing2D_SAC(run_config)
zd = ZData()

# ---Kill portion of the z-sample data if required---
zd.kill_data(params.z_kill_list)

blocks = utils.generate_Ising2D_block_list(max(params.spin_list), params.z_kill_list)

cft = Ising2D_SAC(blocks, params, zd)

spin_list = np.array([0, 0, 2, 2, 2, 4, 4, 6, 6, 8, 10])
delta_teor = np.array([4., 8., 2., 6., 10., 4., 8., 6., 10., 8., 10.])
lambda_teor = np.array([1., 1/100., 1., 1/10., 1/1260., 1/10., 1/126., 1/126., 1/1716., 1/1716., 1/24310.])

for delta_max in range(2, 12, 2):
    mask = np.argwhere(delta_teor <= delta_max)
    spins = spin_list[mask]
    deltas = delta_teor[mask]
    lambdas = lambda_teor[mask]
    reward = LA.norm(cft.compute_ising2d_vector(deltas, spins, lambdas, zd.z, zd.z_conj, params.delta_model))
    print(f'Delta max: {delta_max}, squared mean of crossing equations errors: {reward}')
