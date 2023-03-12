from os import listdir
from os.path import isfile, join
import csv
import numpy as np
from environment.utils import output_to_file
from ope_bounds_BPS import bounds_OPE1, bounds_OPE2, bounds_OPE3
from matplotlib import pyplot as plt
import os
import seaborn as sns

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

path = join('.', 'results_BPS', 'results_BPS_1000_10')
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
ope1_vals = np.zeros((3,tries_per_mode))
ope2_vals = np.zeros((3,tries_per_mode))
ope3_vals = np.zeros((3,tries_per_mode))


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
            
            ope1_vals[integral_mode, j] = ope1
            ope2_vals[integral_mode, j] = ope2
            ope3_vals[integral_mode, j] = ope3
            
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
        output_to_file(file_name=join('BPS_analyzed_modes','integral_mode_'+str(integral_mode)+'.csv'), output=output)
    
    rewards[integral_mode] = rew_coll
    crossing_norms[integral_mode] = crossing_coll
    
    with open(join('BPS_analyzed_modes', 'best_runs.txt'), 'a') as f:
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


### Best 10 rewards
rew_ordered_0 = list(reversed(np.argsort(rewards[0,:])[:rew_to_take]))
rew_ordered_1 = list(reversed(np.argsort(rewards[1,:])[:rew_to_take]))
rew_ordered_2 = list(reversed(np.argsort(rewards[2,:])[:rew_to_take]))

print(rew_ordered_0)
### Average plot
# Initialize the figure
f, ax = plt.subplots()
# Show each observation with a scatterplot
sns.stripplot(
    x=0*np.ones(rew_to_take), y=crossing_norms[0][rew_ordered_0], color='blue',
    dodge=True, alpha=.25, zorder=1, legend=False
)

sns.stripplot(
    x=1*np.ones(rew_to_take), y=crossing_norms[1][rew_ordered_1], color='blue',
    dodge=True, alpha=.25, zorder=1, legend=False
)

sns.stripplot(
    x=2*np.ones(rew_to_take), y=crossing_norms[2][rew_ordered_2], color='blue',
    dodge=True, alpha=.25, zorder=1, legend=False
)
# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(
    x=[0,1,2], y=[np.mean(crossing_norms[0][rew_ordered_0]),np.mean(crossing_norms[1][rew_ordered_1]),np.mean(crossing_norms[2][rew_ordered_2])], color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)

plt.legend()
# Improve the legend
sns.move_legend(
    ax, loc="upper right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
)
plt.xlabel('Number of integral constraints')
plt.ylabel('Crossing equations norm')
plt.title(f'Crossing equations norm with different integral constraints, {rew_to_take} best runs')
plt.savefig(join(f'BPS_analyzed_modes', f'crossing_norms.jpg'), dpi=300)
#plt.show()
plt.close()


### Average plot
# Initialize the figure
f, ax = plt.subplots()
# Show each observation with a scatterplot
sns.stripplot(
    x=1*np.ones(rew_to_take), y=constraints1[1][rew_ordered_1], color='blue',
    dodge=True, alpha=.25, zorder=1, legend=False
)

sns.stripplot(
    x=2*np.ones(rew_to_take), y=constraints1[2][rew_ordered_2], color='blue',
    dodge=True, alpha=.25, zorder=1, legend=False
)
# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(
    x=[1], y=[np.mean(constraints1[1][rew_ordered_1])], color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)
sns.pointplot(
    x=[2], y=[np.mean(constraints1[2][rew_ordered_2])], color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)

plt.legend()
# Improve the legend
sns.move_legend(
    ax, loc="upper right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
)
plt.xlabel('Number of integral constraints')
plt.ylabel('Integral constraint 1')
plt.title(f'Integral constraint 1 with different integral constraints, {rew_to_take} best runs')
plt.savefig(join(f'BPS_analyzed_modes', f'constraint1.jpg'), dpi=300)
#plt.show()
plt.close()


### Average plot
# Initialize the figure
f, ax = plt.subplots()
# Show each observation with a scatterplot
sns.stripplot(
    x=2*np.ones(rew_to_take), y=constraints2[2][rew_ordered_2], color='blue',
    dodge=True, alpha=.25, zorder=1, legend=False
)
# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(
    x=[1], y=[np.mean(constraints2[1][rew_ordered_1])], color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)
sns.pointplot(
    x=[2], y=[np.mean(constraints2[2][rew_ordered_2])], color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)

plt.legend()
# Improve the legend
sns.move_legend(
    ax, loc="upper right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
)
plt.xlabel('Number of integral constraints')
plt.ylabel('Integral constraint 2')
plt.title(f'Integral constraint 2, {rew_to_take} best runs')
plt.savefig(join(f'BPS_analyzed_modes', f'constraint2.jpg'), dpi=300)
#plt.show()
plt.close()



### OPE 1 error
### Average plot
# Initialize the figure
f, ax = plt.subplots()
# Show each observation with a scatterplot
sns.stripplot(
    x=0*np.ones(rew_to_take), y=ope1_err[0][rew_ordered_0], color='blue',
    dodge=True, alpha=.25, zorder=1, legend=False
)

sns.stripplot(
    x=1*np.ones(rew_to_take), y=ope1_err[1][rew_ordered_1], color='blue',
    dodge=True, alpha=.25, zorder=1, legend=False
)

sns.stripplot(
    x=2*np.ones(rew_to_take), y=ope1_err[2][rew_ordered_2], color='blue',
    dodge=True, alpha=.25, zorder=1, legend=False
)
# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(
    x=[0], y=[np.mean(ope1_err[0][rew_ordered_0])], color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)
sns.pointplot(
    x=[1], y=[np.mean(ope1_err[1][rew_ordered_1])], color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)
sns.pointplot(
    x=[2], y=[np.mean(ope1_err[2][rew_ordered_2])], color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)

plt.legend()
# Improve the legend
sns.move_legend(
    ax, loc="upper right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
)
plt.xlabel('Number of integral constraints')
plt.ylabel('Relative error on OPE 1')
plt.title(f'OPE coefficient 1 error with different integral modes, {rew_to_take} best runs')
plt.savefig(join(f'BPS_analyzed_modes', f'ope1_err.jpg'), dpi=300)
#plt.show()
plt.close()


### OPE 2 error
### Average plot
# Initialize the figure
f, ax = plt.subplots()
# Show each observation with a scatterplot
sns.stripplot(
    x=0*np.ones(rew_to_take), y=ope2_err[0][rew_ordered_0], color='blue',
    dodge=True, alpha=.25, zorder=1, legend=False
)

sns.stripplot(
    x=1*np.ones(rew_to_take), y=ope2_err[1][rew_ordered_1], color='blue',
    dodge=True, alpha=.25, zorder=1, legend=False
)

sns.stripplot(
    x=2*np.ones(rew_to_take), y=ope2_err[2][rew_ordered_2], color='blue',
    dodge=True, alpha=.25, zorder=1, legend=False
)
# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(
    x=[0], y=[np.mean(ope2_err[0][rew_ordered_0])], color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)
sns.pointplot(
    x=[1], y=[np.mean(ope2_err[1][rew_ordered_1])], color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)
sns.pointplot(
    x=[2], y=[np.mean(ope2_err[2][rew_ordered_2])], color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)

plt.legend()
# Improve the legend
sns.move_legend(
    ax, loc="upper right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
)
plt.xlabel('Number of integral constraints')
plt.ylabel('Relative error on OPE 2')
plt.title(f'OPE coefficient 2 error with different integral modes, {rew_to_take} best runs')
plt.savefig(join(f'BPS_analyzed_modes', f'ope2_err.jpg'), dpi=300)
#plt.show()
plt.close()


### OPE 3 error
### Average plot
# Initialize the figure
f, ax = plt.subplots()
# Show each observation with a scatterplot
sns.stripplot(
    x=0*np.ones(rew_to_take), y=ope3_err[0][rew_ordered_0], color='blue',
    dodge=True, alpha=.25, zorder=1, legend=False
)

sns.stripplot(
    x=1*np.ones(rew_to_take), y=ope3_err[1][rew_ordered_1], color='blue',
    dodge=True, alpha=.25, zorder=1, legend=False
)

sns.stripplot(
    x=2*np.ones(rew_to_take), y=ope3_err[2][rew_ordered_2], color='blue',
    dodge=True, alpha=.25, zorder=1, legend=False
)
# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(
    x=[0], y=[np.mean(ope3_err[0][rew_ordered_0])], color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)
sns.pointplot(
    x=[1], y=[np.mean(ope3_err[1][rew_ordered_1])], color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)
sns.pointplot(
    x=[2], y=[np.mean(ope3_err[2][rew_ordered_2])], color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)

plt.legend()
# Improve the legend
sns.move_legend(
    ax, loc="upper right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
)
plt.xlabel('Number of integral constraints')
plt.ylabel('Relative error on OPE 3')
plt.title(f'OPE coefficient 3 error with different integral modes, {rew_to_take} best runs')
plt.savefig(join(f'BPS_analyzed_modes', f'ope3_err.jpg'), dpi=300)
#plt.show()
plt.close()


### OPE 1 values
### Average plot
# Initialize the figure
f, ax = plt.subplots()
# Show each observation with a scatterplot
plt.fill_between(x=[0,1,2], y1=bounds_OPE1[g_index, 0], y2=bounds_OPE1[g_index, 1], color='green', alpha=0.2)
sns.stripplot(
    x=0*np.ones(rew_to_take), y=ope1_vals[0][rew_ordered_0], color='blue',
    dodge=True, alpha=.25, zorder=1, legend=False
)

sns.stripplot(
    x=1*np.ones(rew_to_take), y=ope1_vals[1][rew_ordered_1], color='blue',
    dodge=True, alpha=.25, zorder=1, legend=False
)

sns.stripplot(
    x=2*np.ones(rew_to_take), y=ope1_vals[2][rew_ordered_2], color='blue',
    dodge=True, alpha=.25, zorder=1, legend=False
)
# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(
    x=[0], y=[np.mean(ope1_vals[0][rew_ordered_0])], color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)
sns.pointplot(
    x=[1], y=[np.mean(ope1_vals[1][rew_ordered_1])], color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)
sns.pointplot(
    x=[2], y=[np.mean(ope1_vals[2][rew_ordered_2])], color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)

plt.legend()
# Improve the legend
sns.move_legend(
    ax, loc="upper right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
)

plt.xlabel('Number of integral constraints')
plt.ylabel('Relative error on OPE 3')
plt.title(f'OPE coefficient 3 error with different integral modes, {rew_to_take} best runs')
plt.savefig(join(f'BPS_analyzed_modes', f'ope1_vals.jpg'), dpi=300)
#plt.show()
plt.close()


### OPE 2 values
### Average plot
# Initialize the figure
f, ax = plt.subplots()
# Show each observation with a scatterplot
plt.fill_between(x=[0,1,2], y1=bounds_OPE2[g_index, 0], y2=bounds_OPE2[g_index, 1], color='green', alpha=0.2)
sns.stripplot(
    x=0*np.ones(rew_to_take), y=ope2_vals[0][rew_ordered_0], color='blue',
    dodge=True, alpha=.25, zorder=1, legend=False
)

sns.stripplot(
    x=1*np.ones(rew_to_take), y=ope2_vals[1][rew_ordered_1], color='blue',
    dodge=True, alpha=.25, zorder=1, legend=False
)

sns.stripplot(
    x=2*np.ones(rew_to_take), y=ope2_vals[2][rew_ordered_2], color='blue',
    dodge=True, alpha=.25, zorder=1, legend=False
)
# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(
    x=[0], y=[np.mean(ope2_vals[0][rew_ordered_0])], color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)
sns.pointplot(
    x=[1], y=[np.mean(ope2_vals[1][rew_ordered_1])], color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)
sns.pointplot(
    x=[2], y=[np.mean(ope2_vals[2][rew_ordered_2])], color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)

plt.legend()
# Improve the legend
sns.move_legend(
    ax, loc="upper right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
)

plt.xlabel('Number of integral constraints')
plt.ylabel('Relative error on OPE 3')
plt.title(f'OPE coefficient 3 error with different integral modes, {rew_to_take} best runs')
plt.savefig(join(f'BPS_analyzed_modes', f'ope2_vals.jpg'), dpi=300)
#plt.show()
plt.close()


### OPE 3 values
### Average plot
# Initialize the figure
f, ax = plt.subplots()
# Show each observation with a scatterplot
plt.fill_between(x=[0,1,2], y1=bounds_OPE3[g_index, 0], y2=bounds_OPE3[g_index, 1], color='green', alpha=0.2)
sns.stripplot(
    x=0*np.ones(rew_to_take), y=ope3_vals[0][rew_ordered_0], color='blue',
    dodge=True, alpha=.25, zorder=1, legend=False
)

sns.stripplot(
    x=1*np.ones(rew_to_take), y=ope3_vals[1][rew_ordered_1], color='blue',
    dodge=True, alpha=.25, zorder=1, legend=False
)

sns.stripplot(
    x=2*np.ones(rew_to_take), y=ope3_vals[2][rew_ordered_2], color='blue',
    dodge=True, alpha=.25, zorder=1, legend=False
)
# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(
    x=[0], y=[np.mean(ope3_vals[0][rew_ordered_0])], color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)
sns.pointplot(
    x=[1], y=[np.mean(ope3_vals[1][rew_ordered_1])], color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)
sns.pointplot(
    x=[2], y=[np.mean(ope3_vals[2][rew_ordered_2])], color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)

plt.legend()
# Improve the legend
sns.move_legend(
    ax, loc="upper right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
)

plt.xlabel('Number of integral constraints')
plt.ylabel('Relative error on OPE 3')
plt.title(f'OPE coefficient 3 error with different integral modes, {rew_to_take} best runs')
plt.savefig(join(f'BPS_analyzed_modes', f'ope3_vals.jpg'), dpi=300)
#plt.show()
plt.close()


