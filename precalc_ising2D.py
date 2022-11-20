from environment.blocks_ising2D import Ising2D_SAC
from parameters_ising2D import ParametersIsing2D_SAC
from environment.data_z_sample import ZData
from environment import utils
import numpy as np

run_config = {}
run_config['faff_max'] = 10000
run_config['pc_max'] = 5
run_config['window_rate'] = 0.5
run_config['max_window_exp'] = 20
run_config['same_spin_hierarchy'] = True
run_config['dyn_shift'] = 0.5
run_config['reward_scale'] = 0.001


params = ParametersIsing2D_SAC(run_config)
zd = ZData()
# ---Kill portion of the z-sample data if required---
zd.kill_data(params.z_kill_list)
cft = Ising2D_SAC(block_list=[], params=params, z_data=zd)

delta_start = params.delta_start
delta_end_increment = params.delta_end_increment
delta_sep = params.delta_sep

max_spin=10
for i in range(0, max_spin + 2, 2):
    ell = i//2
    tmp_name = 'block_lattices/ising2D_blocks_spin_sigma' + str(i) + '.csv'
    start = delta_start[ell]
    deltas = np.arange(start=start, stop=start+delta_end_increment, step=delta_sep)
    print(deltas)
    values = []
    for delta in deltas:
        value = cft.ising2d_precalc(np.array([delta]), np.array([i]), cft.z, cft.z_conj, cft.delta_model).real
        utils.output_to_file(file_name=tmp_name, output=value)