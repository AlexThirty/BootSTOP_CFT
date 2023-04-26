from os import listdir
from os.path import isfile, join
import csv
import numpy as np
from environment.utils import output_to_file
from ope_bounds_BPS import bounds_OPE1, bounds_OPE2, bounds_OPE3
from matplotlib import pyplot as plt
import itertools
import os
import re

gs = np.concatenate((np.arange(start=0.01, stop=0.25, step=0.01),
                     np.arange(start=0.25, stop=4.05, step=0.05),
                     #np.arange(start=4.25, stop=5.25, step=0.25)
                     ))
gs = np.around(gs, decimals=2)
g = 1.
g_index = np.argwhere(gs==g)[0]


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

path = join('.', 'results_BPS', 'results_BPS_grid_sum')
save_path = join('.', 'BPS_analyzed_grid_sum')
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
r = re.compile('sac[0-9]+.csv')
onlyfiles = list(filter(r.match, onlyfiles))
#print(onlyfiles)
best_reward = 0.

#w1s = [0.0001, 0.0005, 0.001, 0.005]
#w2s = [0.001, 0.01, 0.1, 1., 10.]
#w1s = [0.1, 1., 10., 100., 1000., 10000., 100000., 1e6]
#w2s = [0.1, 1., 10., 100., 1000., 10000., 100000., 1e6]
w1s = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
w2s = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

grid = list(itertools.product(w1s, w2s))
grid_pts = len(grid)
tries_per_mode = 10
delta_len = 10
lambda_len = 10

rewards = np.zeros((grid_pts, tries_per_mode))
ope1_err = np.zeros((grid_pts, tries_per_mode))
ope2_err = np.zeros((grid_pts, tries_per_mode))
ope3_err = np.zeros((grid_pts, tries_per_mode))
crossing_norms = np.zeros((grid_pts, tries_per_mode))
constraints1 = np.zeros((grid_pts, tries_per_mode))
constraints2 = np.zeros((grid_pts, tries_per_mode))


for i in range(grid_pts):
    best_reward = 0.
    rew_coll = []
    deltas_coll = []
    lambdas_coll = []
    lambdas_err = []
    crossing_coll = []
    constraint1_coll = []
    constraint2_coll = []
    w1 = grid[i][0]
    w2 = grid[i][1]
    if not os.path.exists(join(save_path, f'w1_{w1}_w2_{w2}')):
        os.makedirs(join(save_path, f'w1_{w1}_w2_{w2}'))
    for j in range(tries_per_mode):
        currf = open(join(path, 'sac'+str(i*tries_per_mode+j)+'.csv'))
        csv_raw = csv.reader(currf)
        sp = list(csv_raw)
        data = sp[-2]
        
        if len(data)>10:
            curr_rew = float(data[1])
            rew_coll.append(curr_rew)
            
            curr_crossing = float(data[2])
            crossing_coll.append(curr_crossing)
            curr_constraint1 = float(data[3])
            constraint1_coll.append(curr_constraint1)
            curr_constraint2 = float(data[4])
            constraint2_coll.append(curr_constraint2)
            
            curr_delta = [float(data[i]) for i in range(5, 5+delta_len)]
            deltas_coll.append(curr_delta)
            print(f'w1={w1}, w2={w2}')
            print(j)
            curr_lambda = [float(data[i]) for i in range(5+delta_len, 5+delta_len+lambda_len)]
            lambdas_coll.append(curr_lambda)
            
            ope1 = curr_lambda[0]
            ope2 = curr_lambda[1]
            ope3 = curr_lambda[2]
            
            err1 = get_lambda_error(ope1, 1, g_index)
            err2 = get_lambda_error(ope2, 2, g_index)
            err3 = get_lambda_error(ope3, 3, g_index)
            
            ope1_err[i, j] = err1
            ope2_err[i, j] = err2
            ope3_err[i, j] = err3
            
            
            if curr_rew > best_reward:
                best_run = float(data[0])
                best_reward = curr_rew
                deltas = curr_delta
                lambdas = curr_lambda
                
        currf.close()
    orderer = np.argsort(rew_coll)
    for el in reversed(orderer):
        
        output = np.concatenate(([el], [rew_coll[el]], [ope1_err[i, el]], [ope2_err[i, el]], [ope3_err[i, el]], [crossing_coll[el]], [constraint1_coll[el]], [constraint2_coll[el]], deltas_coll[el], lambdas_coll[el]))
        output_to_file(file_name=join(save_path,f'w1_{w1}_w2_{w2}.csv'), output=output)
    
    rewards[i] = rew_coll
    crossing_norms[i] = crossing_coll
    constraints1[i] = constraint1_coll
    constraints2[i] = constraint2_coll
    
    # Best by reward
    orderer = np.argsort(rew_coll)
    with open(join(save_path, 'best_rew.txt'), 'a') as f:
        print(f'w1={w1}, w2={w2}', file=f)
        print(f'Best run: {best_run}', file=f)
        print(f'Reward: {rew_coll[orderer[-1]]}', file=f)
        print(f'OPE1 error: {ope1_err[i, orderer[-1]]}', file=f)
        print(f'OPE2 error: {ope2_err[i, orderer[-1]]}', file=f)
        print(f'OPE3 error: {ope3_err[i, orderer[-1]]}', file=f)
        print(f'Crossing norms: {crossing_coll[orderer[-1]]}', file=f)
        print(f'Constraint 1: {constraint1_coll[orderer[-1]]}', file=f)
        print(f'Constraint 2: {constraint2_coll[orderer[-1]]}', file=f)
        print(f'Deltas: {deltas_coll[orderer[-1]]}', file=f)
        print(f'Lambdas: {lambdas_coll[orderer[-1]]}', file=f)
        
    rew_ordered = np.flip(np.argsort(rewards[i,:])[-10:])

    plt.figure(figsize=(6, 6))
    plt.scatter(range(10), crossing_norms[i, rew_ordered], c=range(10), cmap='tab10')
    plt.title('Crossing equation norms (best rew)')
    plt.xlabel('Best 10 tries')
    plt.ylabel('Crossing equation vector norm')
    plt.yscale('log')
    #plt.legend(fontsize=5)
    plt.savefig(join(save_path, f'w1_{w1}_w2_{w2}','rew_crossing_norms.jpg'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(range(10), constraints1[i, rew_ordered], c=range(10), cmap='tab10')
    plt.title('Constraint 1 norm (best rew)')
    plt.xlabel('Best 10 tries')
    plt.ylabel('constraint1 norm')
    plt.yscale('log')
    #plt.legend(fontsize=5)
    plt.savefig(join(save_path, f'w1_{w1}_w2_{w2}','rew_constraint1.jpg'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(range(10), constraints2[i, rew_ordered], c=range(10), cmap='tab10')
    plt.title('Constraint 2 norm (best rew)')
    plt.xlabel('Best 10 tries')
    plt.ylabel('constraint2 norm')
    plt.yscale('log')
    #plt.legend(fontsize=5)
    plt.savefig(join(save_path, f'w1_{w1}_w2_{w2}','rew_constraint2.jpg'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(range(10), ope1_err[i, rew_ordered], c=range(10), cmap='tab10')
    plt.title('Error on $C^2_1$ (best reward)')
    plt.xlabel('Best 10 tries')
    plt.ylabel('Error')
    plt.yscale('log')
    #plt.legend(fontsize=5)
    plt.savefig(join(save_path, f'w1_{w1}_w2_{w2}','rew_ope1_err.jpg'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(range(10), ope2_err[i, rew_ordered], c=range(10), cmap='tab10')
    plt.title('Error on $C^2_2$ (best reward)')
    plt.xlabel('Best 10 tries')
    plt.ylabel('Error')
    plt.yscale('log')
    #plt.legend(fontsize=5)
    plt.savefig(join(save_path, f'w1_{w1}_w2_{w2}','rew_ope2_err.jpg'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(range(10), ope3_err[i, rew_ordered], c=range(10), cmap='tab10')
    plt.title('Error on OPE3 coefficient (best rew)')
    plt.xlabel('Best 10 tries')
    plt.ylabel('Error')
    plt.yscale('log')
    #plt.legend(fontsize=5)
    plt.savefig(join(save_path, f'w1_{w1}_w2_{w2}','rew_ope3_err.jpg'), dpi=300)
    plt.close()
    
    # Best by OPE1
    orderer = np.argsort(ope1_err[i,:])
    with open(join(save_path, 'best_ope1.txt'), 'a') as f:
        print(f'w1={w1}, w2={w2}', file=f)
        print(f'Best run: {best_run}', file=f)
        print(f'Reward: {rew_coll[orderer[0]]}', file=f)
        print(f'OPE1 error: {ope1_err[i, orderer[0]]}', file=f)
        print(f'OPE2 error: {ope2_err[i, orderer[0]]}', file=f)
        print(f'OPE3 error: {ope3_err[i, orderer[0]]}', file=f)
        print(f'Crossing norms: {crossing_coll[orderer[0]]}', file=f)
        print(f'Constraint 1: {constraint1_coll[orderer[0]]}', file=f)
        print(f'Constraint 2: {constraint2_coll[orderer[0]]}', file=f)
        print(f'Deltas: {deltas_coll[orderer[0]]}', file=f)
        print(f'Lambdas: {lambdas_coll[orderer[0]]}', file=f)
        
    ope1_ordered = np.argsort(ope1_err[i,:])[:10]

    plt.figure(figsize=(6, 6))
    plt.scatter(range(10), crossing_norms[i, ope1_ordered], c=range(10), cmap='tab10')
    plt.title('Crossing equation norms (best OPE1)')
    plt.xlabel('Best 10 tries')
    plt.ylabel('Crossing equation vector norm')
    plt.yscale('log')
    #plt.legend(fontsize=5)
    plt.savefig(join(save_path, f'w1_{w1}_w2_{w2}','OPE1_crossing_norms.jpg'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(range(10), constraints1[i, ope1_ordered], c=range(10), cmap='tab10')
    plt.title('Constraint 1 norm (best OPE1)')
    plt.xlabel('Best 10 tries')
    plt.ylabel('constraint1 norm')
    plt.yscale('log')
    #plt.legend(fontsize=5)
    plt.savefig(join(save_path, f'w1_{w1}_w2_{w2}','OPE1_constraint1.jpg'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(range(10), constraints2[i, ope1_ordered], c=range(10), cmap='tab10')
    plt.title('Constraint 2 norm (best OPE1)')
    plt.xlabel('Best 10 tries')
    plt.ylabel('constraint2 norm')
    plt.yscale('log')
    #plt.legend(fontsize=5)
    plt.savefig(join(save_path, f'w1_{w1}_w2_{w2}','OPE1_constraint2.jpg'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(range(10), ope1_err[i, ope1_ordered], c=range(10), cmap='tab10')
    plt.title('Error on OPE1 coefficient (best OPE1)')
    plt.xlabel('Best 10 tries')
    plt.ylabel('Error')
    plt.yscale('log')
    #plt.legend(fontsize=5)
    plt.savefig(join(save_path, f'w1_{w1}_w2_{w2}','OPE1_ope1_err.jpg'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(range(10), ope2_err[i, ope1_ordered], c=range(10), cmap='tab10')
    plt.title('Error on OPE2 coefficient (best OPE1)')
    plt.xlabel('Best 10 tries')
    plt.ylabel('Error')
    plt.yscale('log')
    #plt.legend(fontsize=5)
    plt.savefig(join(save_path, f'w1_{w1}_w2_{w2}','OPE1_ope2_err.jpg'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(range(10), ope3_err[i, ope1_ordered], c=range(10), cmap='tab10')
    plt.title('Error on OPE3 coefficient (best OPE1)')
    plt.xlabel('Best 10 tries')
    plt.ylabel('Error')
    plt.yscale('log')
    #plt.legend(fontsize=5)
    plt.savefig(join(save_path, f'w1_{w1}_w2_{w2}','OPE1_ope3_err.jpg'), dpi=300)
    plt.close()
    
    
    # Best by OPE2
    orderer = np.argsort(ope2_err[i,:])
    with open(join(save_path, 'best_ope2.txt'), 'a') as f:
        print(f'w1={w1}, w2={w2}', file=f)
        print(f'Best run: {best_run}', file=f)
        print(f'Reward: {rew_coll[orderer[0]]}', file=f)
        print(f'OPE1 error: {ope1_err[i, orderer[0]]}', file=f)
        print(f'OPE2 error: {ope2_err[i, orderer[0]]}', file=f)
        print(f'OPE3 error: {ope3_err[i, orderer[0]]}', file=f)
        print(f'Crossing norms: {crossing_coll[orderer[0]]}', file=f)
        print(f'Constraint 1: {constraint1_coll[orderer[0]]}', file=f)
        print(f'Constraint 2: {constraint2_coll[orderer[0]]}', file=f)
        print(f'Deltas: {deltas_coll[orderer[0]]}', file=f)
        print(f'Lambdas: {lambdas_coll[orderer[0]]}', file=f)
        
    ope2_ordered = np.argsort(ope2_err[i,:])[:10]

    plt.figure(figsize=(6, 6))
    plt.scatter(range(10), crossing_norms[i, ope2_ordered], c=range(10), cmap='tab10')
    plt.title('Crossing equation norms (best OPE2)')
    plt.xlabel('Best 10 tries')
    plt.ylabel('Crossing equation vector norm')
    plt.yscale('log')
    #plt.legend(fontsize=5)
    plt.savefig(join(save_path, f'w1_{w1}_w2_{w2}','OPE2_crossing_norms.jpg'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(range(10), constraints1[i, ope2_ordered], c=range(10), cmap='tab10')
    plt.title('Constraint 1 norm (best OPE2)')
    plt.xlabel('Best 10 tries')
    plt.ylabel('constraint1 norm')
    plt.yscale('log')
    #plt.legend(fontsize=5)
    plt.savefig(join(save_path, f'w1_{w1}_w2_{w2}','OPE2_constraint1.jpg'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(range(10), constraints2[i, ope2_ordered], c=range(10), cmap='tab10')
    plt.title('Constraint 2 norm (best OPE2)')
    plt.xlabel('Best 10 tries')
    plt.ylabel('constraint2 norm')
    plt.yscale('log')
    #plt.legend(fontsize=5)
    plt.savefig(join(save_path, f'w1_{w1}_w2_{w2}','OPE2_constraint2.jpg'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(range(10), ope1_err[i, ope2_ordered], c=range(10), cmap='tab10')
    plt.title('Error on OPE1 coefficient (best OPE2)')
    plt.xlabel('Best 10 tries')
    plt.ylabel('Error')
    plt.yscale('log')
    #plt.legend(fontsize=5)
    plt.savefig(join(save_path, f'w1_{w1}_w2_{w2}','OPE2_ope1_err.jpg'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(range(10), ope2_err[i, ope2_ordered], c=range(10), cmap='tab10')
    plt.title('Error on OPE2 coefficient (best OPE2)')
    plt.xlabel('Best 10 tries')
    plt.ylabel('Error')
    plt.yscale('log')
    #plt.legend(fontsize=5)
    plt.savefig(join(save_path, f'w1_{w1}_w2_{w2}','OPE2_ope2_err.jpg'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(range(10), ope3_err[i, ope2_ordered], c=range(10), cmap='tab10')
    plt.title('Error on OPE3 coefficient (best OPE2)')
    plt.xlabel('Best 10 tries')
    plt.ylabel('Error')
    plt.yscale('log')
    #plt.legend(fontsize=5)
    plt.savefig(join(save_path, f'w1_{w1}_w2_{w2}','OPE2_ope3_err.jpg'), dpi=300)
    plt.close()
    
    
    # Best by OPE3
    orderer = np.argsort(ope3_err[i,:])
    with open(join(save_path, 'best_ope3.txt'), 'a') as f:
        print(f'w1={w1}, w2={w2}', file=f)
        print(f'Best run: {best_run}', file=f)
        print(f'Reward: {rew_coll[orderer[0]]}', file=f)
        print(f'OPE1 error: {ope1_err[i, orderer[0]]}', file=f)
        print(f'OPE2 error: {ope2_err[i, orderer[0]]}', file=f)
        print(f'OPE3 error: {ope3_err[i, orderer[0]]}', file=f)
        print(f'Crossing norms: {crossing_coll[orderer[0]]}', file=f)
        print(f'Constraint 1: {constraint1_coll[orderer[0]]}', file=f)
        print(f'Constraint 2: {constraint2_coll[orderer[0]]}', file=f)
        print(f'Deltas: {deltas_coll[orderer[0]]}', file=f)
        print(f'Lambdas: {lambdas_coll[orderer[0]]}', file=f)
        
    ope3_ordered = np.argsort(ope3_err[i,:])[:10]

    plt.figure(figsize=(6, 6))
    plt.scatter(range(10), crossing_norms[i, ope3_ordered], c=range(10), cmap='tab10')
    plt.title('Crossing equation norms (best OPE3)')
    plt.xlabel('Best 10 tries')
    plt.ylabel('Crossing equation vector norm')
    plt.yscale('log')
    #plt.legend(fontsize=5)
    plt.savefig(join(save_path, f'w1_{w1}_w2_{w2}','OPE3_crossing_norms.jpg'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(range(10), constraints1[i, ope3_ordered], c=range(10), cmap='tab10')
    plt.title('Constraint 1 norm (best OPE3)')
    plt.xlabel('Best 10 tries')
    plt.ylabel('constraint1 norm')
    plt.yscale('log')
    #plt.legend(fontsize=5)
    plt.savefig(join(save_path, f'w1_{w1}_w2_{w2}','OPE3_constraint1.jpg'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(range(10), constraints2[i, ope3_ordered], c=range(10), cmap='tab10')
    plt.title('Constraint 2 norm (best OPE3)')
    plt.xlabel('Best 10 tries')
    plt.ylabel('constraint2 norm')
    plt.yscale('log')
    #plt.legend(fontsize=5)
    plt.savefig(join(save_path, f'w1_{w1}_w2_{w2}','OPE3_constraint2.jpg'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(range(10), ope1_err[i, ope3_ordered], c=range(10), cmap='tab10')
    plt.title('Error on OPE1 coefficient (best OPE3)')
    plt.xlabel('Best 10 tries')
    plt.ylabel('Error')
    plt.yscale('log')
    #plt.legend(fontsize=5)
    plt.savefig(join(save_path, f'w1_{w1}_w2_{w2}','OPE3_ope1_err.jpg'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(range(10), ope2_err[i, ope3_ordered], c=range(10), cmap='tab10')
    plt.title('Error on OPE2 coefficient (best OPE3)')
    plt.xlabel('Best 10 tries')
    plt.ylabel('Error')
    plt.yscale('log')
    #plt.legend(fontsize=5)
    plt.savefig(join(save_path, f'w1_{w1}_w2_{w2}','OPE3_ope2_err.jpg'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(range(10), ope3_err[i, ope3_ordered], c=range(10), cmap='tab10')
    plt.title('Error on OPE3 coefficient (best OPE3)')
    plt.xlabel('Best 10 tries')
    plt.ylabel('Error')
    plt.yscale('log')
    #plt.legend(fontsize=5)
    plt.savefig(join(save_path, f'w1_{w1}_w2_{w2}','OPE3_ope3_err.jpg'), dpi=300)
    plt.close()