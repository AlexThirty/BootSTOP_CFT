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

path = join('.', 'results_BPS_1000_10')
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
        output_to_file(file_name=join('BPS_singleg_analyzed','integral_mode_'+str(integral_mode)+'.csv'), output=output)
    
    rewards[integral_mode] = rew_coll
    crossing_norms[integral_mode] = crossing_coll
    
    with open(join('BPS_singleg_analyzed', 'best.txt'), 'a') as f:
        print(f'Integral mode: {integral_mode}', file=f)
        print(f'Best run: {best_run}', file=f)
        print(f'Reward: {rew_coll[orderer[-1]]}', file=f)
        print(f'OPE1 error: {ope1_err[integral_mode, orderer[-1]]}', file=f)
        print(f'OPE2 error: {ope2_err[integral_mode, orderer[-1]]}', file=f)
        print(f'OPE3 error: {ope3_err[integral_mode, orderer[-1]]}', file=f)
        print(f'Crossing norms: {crossing_coll[orderer[-1]]}', file=f)
        if integral_mode == 1:
            print(f'Constraint 1: {constraint1_coll[orderer[-1]]}', file=f)
        if integral_mode == 2:
            print(f'Constraint 1: {constraint1_coll[orderer[-1]]}', file=f)
            print(f'Constraint 2: {constraint2_coll[orderer[-1]]}', file=f)
        print(f'Deltas: {deltas_coll[orderer[-1]]}', file=f)
        print(f'Lambdas: {lambdas_coll[orderer[-1]]}', file=f)





plt.figure(figsize=(6, 6))
for i in range(3):
    plt.scatter(i*np.ones((tries_per_mode)), crossing_norms[i, :])
plt.title('Crossing equation norms (all)')
plt.xlabel('Integral constraints')
plt.ylabel('Crossing equation vector norm')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'crossing_norms.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
for i in range(1, 3):
    plt.scatter(i*np.ones((tries_per_mode)), constraints1[i,:])
plt.title('Absolute value of constraint 1 (all)')
plt.xlabel('Integral constraints')
plt.ylabel('abs(constraint_1)')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'constraint_1.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
for i in range(2, 3):
    plt.scatter(i*np.ones((tries_per_mode)), constraints2[i,:])
plt.title('Absolute value of constraint 2 (all)')
plt.xlabel('Integral constraints')
plt.ylabel('abs(constraint_2)')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'constraint_2.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
for i in range(0, 3):
    plt.scatter(i*np.ones((tries_per_mode)), ope1_err[i,:])
plt.title('Relative error of OPE 1 w.r.t. bounds (all)')
plt.xlabel('Integral constraints')
plt.ylabel('OPE_1 relative error')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'ope1_err.jpg'), dpi=300)
plt.close()


plt.figure(figsize=(6, 6))
for i in range(0, 3):
    plt.scatter(i*np.ones((tries_per_mode)), ope2_err[i,:])
plt.title('Relative error of OPE 2 w.r.t. bounds (all)')
plt.xlabel('Integral constraints')
plt.ylabel('OPE_2 relative error')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'ope2_err.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
for i in range(0, 3):
    plt.scatter(i*np.ones((tries_per_mode)), ope3_err[i,:])
plt.title('Relative error of OPE 3 w.r.t. bounds (all)')
plt.xlabel('Integral constraints')
plt.ylabel('OPE_3 relative error')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'ope3_err.jpg'), dpi=300)
plt.close()


### Best 10 rewards
rew_ordered_0 = np.argsort(rewards[0,:])[-10:]
rew_ordered_1 = np.argsort(rewards[1,:])[-10:]
rew_ordered_2 = np.argsort(rewards[2,:])[-10:]

plt.figure(figsize=(6, 6))    
plt.scatter(0*np.ones((10)), crossing_norms[0, rew_ordered_0], c=range(10), cmap='tab10')
plt.scatter(1*np.ones((10)), crossing_norms[1, rew_ordered_1], c=range(10), cmap='tab10')
plt.scatter(2*np.ones((10)), crossing_norms[2, rew_ordered_2], c=range(10), cmap='tab10')
plt.title('Crossing equation norms (best rew)')
plt.xlabel('Integral constraints')
plt.ylabel('Crossing equation vector norm')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'rew_crossing_norms.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(0*np.ones((10)), constraints1[0, rew_ordered_0], c=range(10), cmap='tab10')
plt.scatter(1*np.ones((10)), constraints1[1, rew_ordered_1], c=range(10), cmap='tab10')
plt.scatter(2*np.ones((10)), constraints1[2, rew_ordered_2], c=range(10), cmap='tab10')
plt.title('Absolute value of constraint 1 (best rew)')
plt.xlabel('Integral constraints')
plt.ylabel('abs(constraint_1)')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'rew_constraint_1.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(0*np.ones((10)), constraints2[0, rew_ordered_0], c=range(10), cmap='tab10')
plt.scatter(1*np.ones((10)), constraints2[1, rew_ordered_1], c=range(10), cmap='tab10')
plt.scatter(2*np.ones((10)), constraints2[2, rew_ordered_2], c=range(10), cmap='tab10')
plt.title('Absolute value of constraint 2 (best rew)')
plt.xlabel('Integral constraints')
plt.ylabel('abs(constraint_2)')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'rew_constraint_2.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(0*np.ones((10)), ope1_err[0, rew_ordered_0], c=range(10), cmap='tab10')
plt.scatter(1*np.ones((10)), ope1_err[1, rew_ordered_1], c=range(10), cmap='tab10')
plt.scatter(2*np.ones((10)), ope1_err[2, rew_ordered_2], c=range(10), cmap='tab10')
plt.title('Relative error of OPE 1 w.r.t. bounds (best rew)')
plt.xlabel('Integral constraints')
plt.ylabel('OPE_1 relative error')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'rew_ope1_err.jpg'), dpi=300)
plt.close()


plt.figure(figsize=(6, 6))
plt.scatter(0*np.ones((10)), ope2_err[0, rew_ordered_0], c=range(10), cmap='tab10')
plt.scatter(1*np.ones((10)), ope2_err[1, rew_ordered_1], c=range(10), cmap='tab10')
plt.scatter(2*np.ones((10)), ope2_err[2, rew_ordered_2], c=range(10), cmap='tab10')
plt.title('Relative error of OPE 2 w.r.t. bounds (best rew)')
plt.xlabel('Integral constraints')
plt.ylabel('OPE_2 relative error')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'rew_ope2_err.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(0*np.ones((10)), ope3_err[0, rew_ordered_0], c=range(10), cmap='tab10')
plt.scatter(1*np.ones((10)), ope3_err[1, rew_ordered_1], c=range(10), cmap='tab10')
plt.scatter(2*np.ones((10)), ope3_err[2, rew_ordered_2], c=range(10), cmap='tab10')
plt.title('Relative error of OPE 3 w.r.t. bounds (best rew)')
plt.xlabel('Integral constraints')
plt.ylabel('OPE_3 relative error')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'rew_ope3_err.jpg'), dpi=300)
plt.close()

### Best OPE1
ope1_ordered_0 = np.argsort(ope1_err[0,:])[:10]
ope1_ordered_1 = np.argsort(ope1_err[1,:])[:10]
ope1_ordered_2 = np.argsort(ope1_err[2,:])[:10]

plt.figure(figsize=(6, 6))    
plt.scatter(0*np.ones((10)), crossing_norms[0, ope1_ordered_0], c=range(10), cmap='tab10')
plt.scatter(1*np.ones((10)), crossing_norms[1, ope1_ordered_1], c=range(10), cmap='tab10')
plt.scatter(2*np.ones((10)), crossing_norms[2, ope1_ordered_2], c=range(10), cmap='tab10')
plt.title('Crossing equation norms (best OPE1)')
plt.xlabel('Integral constraints')
plt.ylabel('Crossing equation vector norm')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'OPE1_crossing_norms.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(0*np.ones((10)), constraints1[0, ope1_ordered_0], c=range(10), cmap='tab10')
plt.scatter(1*np.ones((10)), constraints1[1, ope1_ordered_1], c=range(10), cmap='tab10')
plt.scatter(2*np.ones((10)), constraints1[2, ope1_ordered_2], c=range(10), cmap='tab10')
plt.title('Absolute value of constraint 1 (best OPE1)')
plt.xlabel('Integral constraints')
plt.ylabel('abs(constraint_1)')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'OPE1_constraint_1.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(0*np.ones((10)), constraints2[0, ope1_ordered_0], c=range(10), cmap='tab10')
plt.scatter(1*np.ones((10)), constraints2[1, ope1_ordered_1], c=range(10), cmap='tab10')
plt.scatter(2*np.ones((10)), constraints2[2, ope1_ordered_2], c=range(10), cmap='tab10')
plt.title('Absolute value of constraint 2 (best OPE1)')
plt.xlabel('Integral constraints')
plt.ylabel('abs(constraint_2)')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'OPE1_constraint_2.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(0*np.ones((10)), ope1_err[0, ope1_ordered_0], c=range(10), cmap='tab10')
plt.scatter(1*np.ones((10)), ope1_err[1, ope1_ordered_1], c=range(10), cmap='tab10')
plt.scatter(2*np.ones((10)), ope1_err[2, ope1_ordered_2], c=range(10), cmap='tab10')
plt.title('Relative error of OPE 1 w.r.t. bounds (best OPE1)')
plt.xlabel('Integral constraints')
plt.ylabel('OPE_1 relative error')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'OPE1_ope1_err.jpg'), dpi=300)
plt.close()


plt.figure(figsize=(6, 6))
plt.scatter(0*np.ones((10)), ope2_err[0, ope1_ordered_0], c=range(10), cmap='tab10')
plt.scatter(1*np.ones((10)), ope2_err[1, ope1_ordered_1], c=range(10), cmap='tab10')
plt.scatter(2*np.ones((10)), ope2_err[2, ope1_ordered_2], c=range(10), cmap='tab10')
plt.title('Relative error of OPE 2 w.r.t. bounds (best OPE1)')
plt.xlabel('Integral constraints')
plt.ylabel('OPE_2 relative error')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'OPE1_ope2_err.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(0*np.ones((10)), ope3_err[0, ope1_ordered_0], c=range(10), cmap='tab10')
plt.scatter(1*np.ones((10)), ope3_err[1, ope1_ordered_1], c=range(10), cmap='tab10')
plt.scatter(2*np.ones((10)), ope3_err[2, ope1_ordered_2], c=range(10), cmap='tab10')
plt.title('Relative error of OPE 3 w.r.t. bounds (best OPE1)')
plt.xlabel('Integral constraints')
plt.ylabel('OPE_3 relative error')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'OPE1_ope3_err.jpg'), dpi=300)
plt.close()


ope2_ordered_0 = np.argsort(ope2_err[0,:])[:10]
ope2_ordered_1 = np.argsort(ope2_err[1,:])[:10]
ope2_ordered_2 = np.argsort(ope2_err[2,:])[:10]

plt.figure(figsize=(6, 6))    
plt.scatter(0*np.ones((10)), crossing_norms[0, ope2_ordered_0], c=range(10), cmap='tab10')
plt.scatter(1*np.ones((10)), crossing_norms[1, ope2_ordered_1], c=range(10), cmap='tab10')
plt.scatter(2*np.ones((10)), crossing_norms[2, ope2_ordered_2], c=range(10), cmap='tab10')
plt.title('Crossing equation norms (best OPE2)')
plt.xlabel('Integral constraints')
plt.ylabel('Crossing equation vector norm')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'OPE2_crossing_norms.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(0*np.ones((10)), constraints1[0, ope2_ordered_0], c=range(10), cmap='tab10')
plt.scatter(1*np.ones((10)), constraints1[1, ope2_ordered_1], c=range(10), cmap='tab10')
plt.scatter(2*np.ones((10)), constraints1[2, ope2_ordered_2], c=range(10), cmap='tab10')
plt.title('Absolute value of constraint 1 (best OPE2)')
plt.xlabel('Integral constraints')
plt.ylabel('abs(constraint_1)')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'OPE2_constraint_1.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(0*np.ones((10)), constraints2[0, ope2_ordered_0], c=range(10), cmap='tab10')
plt.scatter(1*np.ones((10)), constraints2[1, ope2_ordered_1], c=range(10), cmap='tab10')
plt.scatter(2*np.ones((10)), constraints2[2, ope2_ordered_2], c=range(10), cmap='tab10')
plt.title('Absolute value of constraint 2 (best OPE2)')
plt.xlabel('Integral constraints')
plt.ylabel('abs(constraint_2)')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'OPE2_constraint_2.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(0*np.ones((10)), ope1_err[0, ope2_ordered_0], c=range(10), cmap='tab10')
plt.scatter(1*np.ones((10)), ope1_err[1, ope2_ordered_1], c=range(10), cmap='tab10')
plt.scatter(2*np.ones((10)), ope1_err[2, ope2_ordered_2], c=range(10), cmap='tab10')
plt.title('Relative error of OPE 1 w.r.t. bounds (best OPE2)')
plt.xlabel('Integral constraints')
plt.ylabel('OPE_1 relative error')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'OPE2_ope1_err.jpg'), dpi=300)
plt.close()


plt.figure(figsize=(6, 6))
plt.scatter(0*np.ones((10)), ope2_err[0, ope2_ordered_0], c=range(10), cmap='tab10')
plt.scatter(1*np.ones((10)), ope2_err[1, ope2_ordered_1], c=range(10), cmap='tab10')
plt.scatter(2*np.ones((10)), ope2_err[2, ope2_ordered_2], c=range(10), cmap='tab10')
plt.title('Relative error of OPE 2 w.r.t. bounds (best OPE2)')
plt.xlabel('Integral constraints')
plt.ylabel('OPE_2 relative error')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'OPE2_ope2_err.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(0*np.ones((10)), ope3_err[0, ope2_ordered_0], c=range(10), cmap='tab10')
plt.scatter(1*np.ones((10)), ope3_err[1, ope2_ordered_1], c=range(10), cmap='tab10')
plt.scatter(2*np.ones((10)), ope3_err[2, ope2_ordered_2], c=range(10), cmap='tab10')
plt.title('Relative error of OPE 3 w.r.t. bounds (best OPE2)')
plt.xlabel('Integral constraints')
plt.ylabel('OPE_3 relative error')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'OPE2_ope3_err.jpg'), dpi=300)
plt.close()


ope3_ordered_0 = np.argsort(ope3_err[0,:])[:10]
ope3_ordered_1 = np.argsort(ope3_err[1,:])[:10]
ope3_ordered_2 = np.argsort(ope3_err[2,:])[:10]

plt.figure(figsize=(6, 6))    
plt.scatter(0*np.ones((10)), crossing_norms[0, ope3_ordered_0], c=range(10), cmap='tab10')
plt.scatter(1*np.ones((10)), crossing_norms[1, ope3_ordered_1], c=range(10), cmap='tab10')
plt.scatter(2*np.ones((10)), crossing_norms[2, ope3_ordered_2], c=range(10), cmap='tab10')
plt.title('Crossing equation norms (best OPE3)')
plt.xlabel('Integral constraints')
plt.ylabel('Crossing equation vector norm')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'OPE3_crossing_norms.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(0*np.ones((10)), constraints1[0, ope3_ordered_0], c=range(10), cmap='tab10')
plt.scatter(1*np.ones((10)), constraints1[1, ope3_ordered_1], c=range(10), cmap='tab10')
plt.scatter(2*np.ones((10)), constraints1[2, ope3_ordered_2], c=range(10), cmap='tab10')
plt.title('Absolute value of constraint 1 (best OPE3)')
plt.xlabel('Integral constraints')
plt.ylabel('abs(constraint_1)')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'OPE3_constraint_1.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(0*np.ones((10)), constraints2[0, ope3_ordered_0], c=range(10), cmap='tab10')
plt.scatter(1*np.ones((10)), constraints2[1, ope3_ordered_1], c=range(10), cmap='tab10')
plt.scatter(2*np.ones((10)), constraints2[2, ope3_ordered_2], c=range(10), cmap='tab10')
plt.title('Absolute value of constraint 2 (best OPE3)')
plt.xlabel('Integral constraints')
plt.ylabel('abs(constraint_2)')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'OPE3_constraint_2.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(0*np.ones((10)), ope1_err[0, ope3_ordered_0], c=range(10), cmap='tab10')
plt.scatter(1*np.ones((10)), ope1_err[1, ope3_ordered_1], c=range(10), cmap='tab10')
plt.scatter(2*np.ones((10)), ope1_err[2, ope3_ordered_2], c=range(10), cmap='tab10')
plt.title('Relative error of OPE 1 w.r.t. bounds (best OPE3)')
plt.xlabel('Integral constraints')
plt.ylabel('OPE_1 relative error')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'OPE3_ope1_err.jpg'), dpi=300)
plt.close()


plt.figure(figsize=(6, 6))
plt.scatter(0*np.ones((10)), ope2_err[0, ope3_ordered_0], c=range(10), cmap='tab10')
plt.scatter(1*np.ones((10)), ope2_err[1, ope3_ordered_1], c=range(10), cmap='tab10')
plt.scatter(2*np.ones((10)), ope2_err[2, ope3_ordered_2], c=range(10), cmap='tab10')
plt.title('Relative error of OPE 2 w.r.t. bounds (best OPE3)')
plt.xlabel('Integral constraints')
plt.ylabel('OPE_2 relative error')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'OPE3_ope2_err.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(0*np.ones((10)), ope3_err[0, ope3_ordered_0], c=range(10), cmap='tab10')
plt.scatter(1*np.ones((10)), ope3_err[1, ope3_ordered_1], c=range(10), cmap='tab10')
plt.scatter(2*np.ones((10)), ope3_err[2, ope3_ordered_2], c=range(10), cmap='tab10')
plt.title('Relative error of OPE 3 w.r.t. bounds (best OPE3)')
plt.xlabel('Integral constraints')
plt.ylabel('OPE_3 relative error')
plt.yscale('log')
#plt.legend(fontsize=5)
plt.savefig(join('BPS_singleg_analyzed', 'OPE3_ope3_err.jpg'), dpi=300)
plt.close()