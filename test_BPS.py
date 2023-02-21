from environment.blocks_BPS import BPS_SAC
from environment.data_z_sample import ZData
from environment import utils
import numpy as np
import values_BPS
from parameters_BPS_hybrid import ParametersBPS_SAC

def get_teor_deltas(g):
    deltas = np.zeros(10)
    deltas[0] = values_BPS.delta1[str(g)]
    deltas[1] = values_BPS.delta2[str(g)]
    deltas[2] = values_BPS.delta3[str(g)]
    deltas[3] = values_BPS.delta4[str(g)]
    deltas[4] = values_BPS.delta5[str(g)]
    deltas[5] = values_BPS.delta6[str(g)]
    deltas[6] = values_BPS.delta7[str(g)]
    deltas[7] = values_BPS.delta8[str(g)]
    deltas[8] = values_BPS.delta9[str(g)]
    deltas[9] = values_BPS.delta10[str(g)]
    return deltas


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
integral_mode = 2
print(gs)
g = 0.1
gs = np.around(gs, decimals=2)
zd = ZData()
params = ParametersBPS_SAC(config=run_config, g=g, integral_mode=integral_mode)
params = ParametersBPS_SAC(config=run_config, g=g, integral_mode=integral_mode)
params_free = ParametersBPS_SAC(config=run_config, g=g, integral_mode=integral_mode)
zd.kill_data(params.z_kill_list)
g_index = np.argwhere(gs==g)[0]
blocks = utils.generate_BPS_block_list(g_index=g_index)
int1_list = utils.generate_BPS_int1_list(g_index=g_index)
int2_list = utils.generate_BPS_int2_list(g_index=g_index)
# ---Load the pre-generated conformal blocks for long multiplets---
#blocks = utils.generate_block_list(max(params.spin_list), params.z_kill_list)
blocks_free = utils.generate_BPS_block_list_free()
int1_list_free = utils.generate_BPS_int1_list_free().reshape((-1))
int2_list_free = utils.generate_BPS_int2_list_free().reshape((-1))
# ---Instantiate the crossing_eqn class---
cft = BPS_SAC(params, zd, blocks, int1_list, int2_list, 1, blocks_free, int1_list_free, int2_list_free)

deltas = np.random.random(5)*10
blocks = []
integrals1 = []
integrals2 = []
for i in range(len(deltas)):
    delta = deltas[i]
    delta = np.clip(delta, a_min=None, a_max=params.delta_start + params.delta_end_increment - params.delta_sep)
    n = int(np.rint((delta - params.delta_start) / params.delta_sep))
    blocks.append(blocks_free[n])
    integrals1.append(int1_list_free[n])
    integrals2.append(int2_list_free[n])

deltas = np.concatenate((get_teor_deltas(g=g), deltas))

for i in range(10):
    lambdas = np.random.random(15)
    a = cft.compute_BPS_vector(deltas, lambdas, cft.chi)
    b = cft.get_hybrid_vector(blocks, lambdas[:10], lambdas[10:])
    print(np.linalg.norm(a-b))
    
for i in range(10):
    lambdas = np.random.random(15)
    a = cft.calc_constraint_1(deltas, lambdas)
    b = cft.get_hybrid_constraint_1(integrals1, lambdas[:10], lambdas[10:])
    print(np.linalg.norm(a-b))
    
for i in range(10):
    lambdas = np.random.random(15)
    a = cft.calc_constraint_2(deltas, lambdas)
    b = cft.get_hybrid_constraint_2(integrals2, lambdas[:10], lambdas[10:])
    print(np.linalg.norm(a-b))