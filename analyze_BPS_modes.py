from os import listdir
from os.path import isfile, join
import csv
import numpy as np
from environment.utils import output_to_file
from ope_bounds_BPS import bounds_OPE1, bounds_OPE2, bounds_OPE3
from matplotlib import pyplot as plt
import os
import seaborn as sns
plt.rcParams.update({'font.size': 12})

gs = np.concatenate((np.arange(start=0.01, stop=0.25, step=0.01),
                     np.arange(start=0.25, stop=4.05, step=0.05),
                     #np.arange(start=4.25, stop=5.25, step=0.25)
                     ))
gs = np.around(gs, decimals=2)
g = 1.
g_index = np.argwhere(gs==g)[0]
rew_to_take = 10

if not os.path.exists(f'BPS_analyzed_modes'):
        os.makedirs(f'BPS_analyzed_modes')

def get_lambda_error(val, ope_index, g_index):
    if ope_index==1:
        lower = bounds_OPE1[g_index, 0]
        upper = bounds_OPE1[g_index, 1]
        half = (lower+upper)/2
    
        if val < lower:
            return abs(val-lower)/lower
        elif val > upper:
            return abs(val-upper)/upper
        else:
            return 0.
    elif ope_index==2:
        lower = bounds_OPE2[g_index, 0]
        upper = bounds_OPE2[g_index, 1]
        half = (lower+upper)/2
        if val < lower:
            return abs(val-lower)/lower
        elif val > upper:
            return abs(val-upper)/upper
        else:
            return 0.
    elif ope_index==3:
        lower = bounds_OPE3[g_index, 0]
        upper = bounds_OPE3[g_index, 1]
        half = (lower+upper)/2
        if val < lower:
            return abs(val-lower)/lower
        elif val > upper:
            return abs(val-upper)/upper
        else:
            return 0.
    else:
        raise ValueError

path = join('.', 'results_BPS', 'results_BPS_0fix_g1_constraints')
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
best_reward = 0.

tries_per_mode = 150
delta_len = 10
lambda_len = 10
best_rew_to_take = 10

rewards = np.zeros((3, tries_per_mode))
crossing_norms = np.zeros((3, tries_per_mode))
constraints1 = np.zeros((3, tries_per_mode))
constraints2 = np.zeros((3, tries_per_mode))
ope1_err = np.zeros((3, tries_per_mode))
ope2_err = np.zeros((3, tries_per_mode))
ope3_err = np.zeros((3, tries_per_mode))
ope_vals = np.zeros((3, tries_per_mode, 10))
ope_means = np.zeros((3, 10))
ope_stds = np.zeros((3, 10))


for integral_mode in range(3):
    best_reward = 0.
    rew_coll = []
    deltas_coll = []
    lambdas_coll = []
    lambdas_err = []
    crossing_coll = []
    constraint1_coll = []
    constraint2_coll = []
    
    for j in range(tries_per_mode):
        currf = open(join(path, 'sac'+str(1000*integral_mode+j)+'.csv'))
        csv_raw = csv.reader(currf)
        sp = list(csv_raw)
        
        last_index = -1
        while len(sp[last_index]) < 5:
            last_index = last_index-1
        data = sp[last_index]

        curr_rew = float(data[1])
        rewards[integral_mode, j] = curr_rew
        
        curr_crossing = float(data[2])
        crossing_norms[integral_mode, j] = curr_crossing
        
        if integral_mode>0:
            curr_constraint1 = float(data[3])
            constraints1[integral_mode,j] = curr_constraint1
        
        if integral_mode == 2:
            curr_constraint2 = float(data[4])
            constraints2[integral_mode,j] = curr_constraint2
        
        curr_delta = [float(data[i]) for i in range(3+integral_mode, 3+integral_mode+delta_len)]
        deltas_coll.append(curr_delta)
        curr_lambda = [float(data[i]) for i in range(3+integral_mode+delta_len, 3+integral_mode+delta_len+lambda_len)]
        lambdas_coll.append(curr_lambda)

        
        ope1 = curr_lambda[0]
        ope2 = curr_lambda[1]
        ope3 = curr_lambda[2]
        
        ope_vals[integral_mode, j, :] = np.array(curr_lambda)
        
        err1 = get_lambda_error(ope1, 1, g_index)
        err2 = get_lambda_error(ope2, 2, g_index)
        err3 = get_lambda_error(ope3, 3, g_index)
        
        ope1_err[integral_mode, j] = err1
        ope2_err[integral_mode, j] = err2
        ope3_err[integral_mode, j] = err3
        
        
        if curr_rew > best_reward:
            best_run = float(data[0])
            best_reward = curr_rew
            deltas = curr_delta
            lambdas = curr_lambda
            
        currf.close()
    orderer = np.flip((np.argsort(rewards[integral_mode,:])))
    
    rewards[integral_mode, :] =  rewards[integral_mode, orderer]
    crossing_norms[integral_mode, :] = crossing_norms[integral_mode, orderer]
    constraints1[integral_mode, :] = constraints1[integral_mode, orderer]
    constraints2[integral_mode, :] = constraints2[integral_mode, orderer]
    ope_vals[integral_mode, :, :] = ope_vals[integral_mode, orderer, :]
    ope1_err[integral_mode, :] = ope1_err[integral_mode, orderer]
    ope2_err[integral_mode, :] = ope2_err[integral_mode, orderer]
    ope3_err[integral_mode, :] = ope3_err[integral_mode, orderer]

    ope_means[integral_mode, :] = np.mean(ope_vals[integral_mode, :best_rew_to_take, :], axis=0)
    ope_stds[integral_mode, :] = np.std(ope_vals[integral_mode, :best_rew_to_take, :], axis=0)
    if integral_mode == 2:
        print(ope_means)
        print(ope_stds)
    
    for el in range(tries_per_mode):
        if integral_mode==0:
            output = np.concatenate(([el], [rewards[integral_mode, el]], [ope1_err[integral_mode, el]], [ope2_err[integral_mode, el]], [ope3_err[integral_mode, el]], [crossing_norms[integral_mode, el]], ope_vals[integral_mode, el]))
        elif integral_mode==1:
            output = np.concatenate(([el], [rewards[integral_mode, el]], [ope1_err[integral_mode, el]], [ope2_err[integral_mode, el]], [ope3_err[integral_mode, el]], [crossing_norms[integral_mode, el]], [constraints1[integral_mode, el]], ope_vals[integral_mode, el]))
        elif integral_mode==2:
            output = np.concatenate(([el], [rewards[integral_mode, el]], [ope1_err[integral_mode, el]], [ope2_err[integral_mode, el]], [ope3_err[integral_mode, el]], [crossing_norms[integral_mode, el]], [constraints1[integral_mode, el]], [constraints2[integral_mode, el]], ope_vals[integral_mode, el]))
        output_to_file(file_name=join('BPS_analyzed_modes','integral_mode_'+str(integral_mode)+'.csv'), output=output)
    
    
    with open(join('BPS_analyzed_modes', 'best_runs.txt'), 'a') as f:
        print(f'Integral mode: {integral_mode}', file=f)
        print(f'Reward: {rewards[integral_mode,:best_rew_to_take]}', file=f)
        print(f'OPE1 error: {ope1_err[integral_mode, :best_rew_to_take]}', file=f)
        print(f'OPE2 error: {ope2_err[integral_mode, :best_rew_to_take]}', file=f)
        print(f'OPE3 error: {ope3_err[integral_mode, :best_rew_to_take]}', file=f)
        print(f'Crossing norms: {crossing_norms[integral_mode, :best_rew_to_take]}', file=f)
        if integral_mode == 1:
            print(f'Constraint 1: {constraints1[integral_mode, :best_rew_to_take]}', file=f)
        if integral_mode == 2:
            print(f'Constraint 1: {constraints1[integral_mode, :best_rew_to_take]}', file=f)
            print(f'Constraint 2: {constraints2[integral_mode, :best_rew_to_take]}', file=f)
        print(f'Lambdas means: {ope_means[integral_mode,  :best_rew_to_take]}', file=f)
        print(f'Lambdas stds: {ope_stds[integral_mode,  :best_rew_to_take]}', file=f)
        print(f'Lambdas rel_errs: {ope_stds[integral_mode,  :best_rew_to_take]/ope_means[integral_mode, :best_rew_to_take]}', file=f)



f, ax = plt.subplots()
plt.scatter(x=0*np.ones(best_rew_to_take), y=ope1_err[0, :best_rew_to_take], color='blue')
plt.scatter(x=1*np.ones(best_rew_to_take), y=ope1_err[1, :best_rew_to_take], color='green')
plt.scatter(x=2*np.ones(best_rew_to_take), y=ope1_err[2, :best_rew_to_take], color='red')
plt.title('Error w.r.t. ground truth, $C_1^2$')
plt.xticks([0, 1, 2], ['No constraints', 'one constraint', 'two constraints'], rotation=0)
plt.savefig(os.path.join('BPS_analyzed_modes', 'ope1_error.png'), dpi=300)
plt.close()


f, ax = plt.subplots()
plt.scatter(x=0*np.ones(best_rew_to_take), y=ope2_err[0, :best_rew_to_take], color='blue')
plt.scatter(x=1*np.ones(best_rew_to_take), y=ope2_err[1, :best_rew_to_take], color='green')
plt.scatter(x=2*np.ones(best_rew_to_take), y=ope2_err[2, :best_rew_to_take], color='red')
plt.title('Error w.r.t. ground truth, $C_2^2$')
plt.xticks([0, 1, 2], ['No constraints', 'one constraint', 'two constraints'], rotation=0)
plt.savefig(os.path.join('BPS_analyzed_modes', 'ope2_error.png'), dpi=300)
plt.close()


f, ax = plt.subplots()
plt.scatter(x=0*np.ones(best_rew_to_take), y=ope3_err[0, :best_rew_to_take], color='blue')
plt.scatter(x=1*np.ones(best_rew_to_take), y=ope3_err[1, :best_rew_to_take], color='green')
plt.scatter(x=2*np.ones(best_rew_to_take), y=ope3_err[2, :best_rew_to_take], color='red')
plt.title('Error w.r.t. ground truth, $C_3^2$')
plt.xticks([0, 1, 2], ['No constraints', 'one constraint', 'two constraints'], rotation=0)
plt.savefig(os.path.join('BPS_analyzed_modes', 'ope3_error.png'), dpi=300)
plt.close()



f, ax = plt.subplots()
plt.scatter(x=0*np.ones(best_rew_to_take), y=crossing_norms[0, :best_rew_to_take], color='blue')
plt.scatter(x=1*np.ones(best_rew_to_take), y=crossing_norms[1, :best_rew_to_take], color='green')
plt.scatter(x=2*np.ones(best_rew_to_take), y=crossing_norms[2, :best_rew_to_take], color='red')
plt.title('$||E(\\Delta, C^2)||$')
plt.xticks([0, 1, 2], ['No constraints', 'one constraint', 'two constraints'], rotation=0)
plt.savefig(os.path.join('BPS_analyzed_modes', 'crossing_norms.png'), dpi=300)
plt.close()

f, ax = plt.subplots()
plt.scatter(x=np.arange(10), y=ope_stds[0, :]/ope_means[0, :], color='blue', label='no constraints')
plt.scatter(x=np.arange(10), y=ope_stds[1, :]/ope_means[1, :], color='green', label='one constraints')
plt.scatter(x=np.arange(10), y=ope_stds[2, :]/ope_means[2, :], color='red', label='two constraints')
plt.legend()
plt.title('Relative errors on $C_n^2$')
plt.xticks(np.arange(10), ['$C_1^2$', '$C_2^2$', '$C_3^2$', '$C_4^2$', '$C_5^2$', '$C_6^2$', '$C_7^2$', '$C_8^2$', '$C_9^2$', '$C_{{10}}^2$'], rotation=0)
plt.savefig(os.path.join('BPS_analyzed_modes', 'rel_errors.png'), dpi=300)
plt.close()

f, ax = plt.subplots()
plt.scatter(x=np.arange(10), y=ope_stds[0, :]/ope_means[0, :], color='blue', label='no constraints')
plt.scatter(x=np.arange(10), y=ope_stds[1, :]/ope_means[1, :], color='green', label='one constraints')
plt.scatter(x=np.arange(10), y=ope_stds[2, :]/ope_means[2, :], color='red', label='two constraints')
plt.legend()
plt.yscale('log')
plt.title('Relative errors on $C_n^2$')
plt.xticks(np.arange(10), ['$C_1^2$', '$C_2^2$', '$C_3^2$', '$C_4^2$', '$C_5^2$', '$C_6^2$', '$C_7^2$', '$C_8^2$', '$C_9^2$', '$C_{{10}}^2$'], rotation=0)
plt.savefig(os.path.join('BPS_analyzed_modes', 'rel_errors_log.png'), dpi=300)
plt.close()

import matplotlib.patches as mpatches

f, ax = plt.subplots()
for i in range(best_rew_to_take):
    plt.scatter(x=np.arange(10), y=ope_vals[0, i, :], color='blue', label='no constraints')
    plt.scatter(x=np.arange(10), y=ope_vals[1, i, :], color='green', label='one constraints')
    plt.scatter(x=np.arange(10), y=ope_vals[2, i, :], color='red', label='two constraints')
red_patch = mpatches.Patch(color='red', label='two constraints')
green_patch = mpatches.Patch(color='green', label='one constraint')
blue_patch = mpatches.Patch(color='blue', label='no constraints')
plt.legend(handles=[blue_patch, green_patch, red_patch])
plt.yscale('log')
plt.title('Values on $C_n^2$')
plt.xticks(np.arange(10), ['$C_1^2$', '$C_2^2$', '$C_3^2$', '$C_4^2$', '$C_5^2$', '$C_6^2$', '$C_7^2$', '$C_8^2$', '$C_9^2$', '$C_{{10}}^2$'], rotation=0)
plt.savefig(os.path.join('BPS_analyzed_modes', 'ope_values.png'), dpi=300)
plt.close()