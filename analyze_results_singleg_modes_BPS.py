from os import listdir
from os.path import isfile, join
import csv
import numpy as np
from environment.utils import output_to_file
from ope_bounds_BPS import bounds_OPE1, bounds_OPE2, bounds_OPE3
from matplotlib import pyplot as plt

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
            return abs(val-half)/half
        elif val > upper:
            return abs(val-half)/half
        else:
            return 0.
    elif ope_index==2:
        lower = bounds_OPE2[g_index, 0]
        upper = bounds_OPE2[g_index, 1]
        half = (lower+upper)/2
        if val < lower:
            return abs(val-half)/half
        elif val > upper:
            return abs(val-half)/half
        else:
            return 0.
    elif ope_index==3:
        lower = bounds_OPE3[g_index, 0]
        upper = bounds_OPE3[g_index, 1]
        half = (lower+upper)/2
        if val < lower:
            return abs(val-half)/half
        elif val > upper:
            return abs(val-half)/half
        else:
            return 0.
    else:
        raise ValueError

path = join('.', 'results')
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
best_reward = 0.

tries_per_mode = 150
delta_len = 10
lambda_len = 10

rewards = np.zeros((3, tries_per_mode))
ope1_err = np.zeros((3, tries_per_mode))
ope2_err = np.zeros((3, tries_per_mode))
ope3_err = np.zeros((3, tries_per_mode))
crossing_norms = np.zeros((3, tries_per_mode))
constraints1 = np.zeros((3, tries_per_mode))
constraints2 = np.zeros((3, tries_per_mode))


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
        data = sp[-1]
        
        if len(data)>10:
            curr_rew = float(data[1])
            rew_coll.append(curr_rew)
            
            curr_crossing = float(data[2])
            crossing_coll.append(curr_crossing)
            if integral_mode>0:
                curr_constraint1 = float(data[3])
                constraint1_coll.append(curr_constraint1)
            
            if integral_mode == 2:
                curr_constraint2 = float(data[4])
                constraint2_coll.append(curr_constraint2)
            
            curr_delta = [float(data[i]) for i in range(3+integral_mode, 3+integral_mode+delta_len)]
            deltas_coll.append(curr_delta)
            print(integral_mode)
            print(j)
            curr_lambda = [float(data[i]) for i in range(3+integral_mode+delta_len, 3+integral_mode+delta_len+lambda_len)]
            lambdas_coll.append(curr_lambda)
            
            ope1 = curr_lambda[0]
            ope2 = curr_lambda[1]
            ope3 = curr_lambda[2]
            
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
    orderer = np.argsort(rew_coll)
    for el in reversed(orderer):
        if integral_mode==0:
            output = np.concatenate(([el], [rew_coll[el]], [ope1_err[integral_mode, el]], [ope2_err[integral_mode, el]], [ope3_err[integral_mode, el]], [crossing_coll[el]], deltas_coll[el], lambdas_coll[el]))
        elif integral_mode==1:
            constraints1[integral_mode] = constraint1_coll
            output = np.concatenate(([el], [rew_coll[el]], [ope1_err[integral_mode, el]], [ope2_err[integral_mode, el]], [ope3_err[integral_mode, el]], [crossing_coll[el]], [constraint1_coll[el]], deltas_coll[el], lambdas_coll[el]))
        elif integral_mode==2:
            constraints1[integral_mode] = constraint1_coll
            constraints2[integral_mode] = constraint2_coll
            output = np.concatenate(([el], [rew_coll[el]], [ope1_err[integral_mode, el]], [ope2_err[integral_mode, el]], [ope3_err[integral_mode, el]], [crossing_coll[el]], [constraint1_coll[el]], [constraint2_coll[el]], deltas_coll[el], lambdas_coll[el]))
        output_to_file(file_name=join('BPS_singleg_analized','integral_mode_'+str(integral_mode)+'.csv'), output=output)
    
    rewards[integral_mode] = rew_coll
    crossing_norms[integral_mode] = crossing_coll
    
    with open(join('BPS_singleg_analized', 'best.txt'), 'a') as f:
        print(f'Integral mode: {integral_mode}', file=f)
        print(f'Best run: {best_run}', file=f)
        print(f'Reward: {rew_coll[orderer[0]]}', file=f)
        print(f'OPE1 error: {ope1_err[integral_mode, orderer[0]]}', file=f)
        print(f'OPE2 error: {ope2_err[integral_mode, orderer[0]]}', file=f)
        print(f'OPE3 error: {ope3_err[integral_mode, orderer[0]]}', file=f)
        print(f'Crossing norms: {crossing_coll[orderer[0]]}', file=f)
        if integral_mode == 1:
            print(f'Constraint 1: {constraint1_coll[orderer[0]]}', file=f)
        if integral_mode == 2:
            print(f'Constraint 1: {constraint1_coll[orderer[0]]}', file=f)
            print(f'Constraint 2: {constraint2_coll[orderer[0]]}', file=f)
        print(f'Deltas: {deltas_coll[orderer[0]]}', file=f)
        print(f'Lambdas: {lambdas_coll[orderer[0]]}', file=f)

plt.figure(figsize=(6, 6))
for i in range(3):
    plt.scatter(i*np.ones((tries_per_mode)), crossing_norms[i, :])
plt.title('Crossing equation norms (all)')
plt.xlabel('Integral constraints')
plt.ylabel('Crossing equation vector norm')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analized', 'crossing_norms.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
for i in range(1, 3):
    plt.scatter(i*np.ones((tries_per_mode)), constraints1[i,:])
plt.title('Absolute value of constraint 1 (all)')
plt.xlabel('Integral constraints')
plt.ylabel('abs(constraint_1)')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analized', 'constraint_1.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
for i in range(2, 3):
    plt.scatter(i*np.ones((tries_per_mode)), constraints2[i,:])
plt.title('Absolute value of constraint 2 (all)')
plt.xlabel('Integral constraints')
plt.ylabel('abs(constraint_2)')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analized', 'constraint_2.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
for i in range(0, 3):
    plt.scatter(i*np.ones((tries_per_mode)), ope1_err[i,:])
plt.title('Relative error of OPE 1 w.r.t. bounds (all)')
plt.xlabel('Integral constraints')
plt.ylabel('OPE_1 relative error')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analized', 'ope1_err.jpg'), dpi=300)
plt.close()


plt.figure(figsize=(6, 6))
for i in range(0, 3):
    plt.scatter(i*np.ones((tries_per_mode)), ope2_err[i,:])
plt.title('Relative error of OPE 2 w.r.t. bounds (all)')
plt.xlabel('Integral constraints')
plt.ylabel('OPE_2 relative error')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analized', 'ope2_err.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
for i in range(0, 3):
    plt.scatter(i*np.ones((tries_per_mode)), ope3_err[i,:])
plt.title('Relative error of OPE 3 w.r.t. bounds (all)')
plt.xlabel('Integral constraints')
plt.ylabel('OPE_3 relative error')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analized', 'ope3_err.jpg'), dpi=300)
plt.close()