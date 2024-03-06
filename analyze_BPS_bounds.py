from os import listdir
from os.path import isfile, join
import csv
import numpy as np
from environment.utils import output_to_file
from ope_bounds_BPS import bounds_OPE1, bounds_OPE2, bounds_OPE3
from matplotlib import pyplot as plt
import values_BPS
import itertools
import os
import re
import seaborn as sns
from values_BPS import *
plt.rcParams.update({'font.size': 16})


best_reward = 0.
delta_len = 10
lambda_len = 10
rew_to_take = 25
OPE_fix = 1
ipopt = False
ipopt_without = False
sac = False

article_g_list = np.array([0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2.])
#article_g_list = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.4, 3.6, 4.0])
#article_g_list = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
#article_g_list = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0])
ipopt_C1_values = np.array([float(C1_IPOPT_with[str(g)]) for g in article_g_list])
ipopt_C2_values = np.array([float(C2_IPOPT_with[str(g)]) for g in article_g_list])
ipopt_C3_values = np.array([float(C3_IPOPT_with[str(g)]) for g in article_g_list])

ipopt_without_C1_values = np.array([float(C1_IPOPT_without[str(g)]) for g in article_g_list])
ipopt_without_C2_values = np.array([float(C2_IPOPT_without[str(g)]) for g in article_g_list])
ipopt_without_C3_values = np.array([float(C3_IPOPT_without[str(g)]) for g in article_g_list])
sac_C1_values = np.array([float(C1_SAC[str(g)]) for g in article_g_list])
sac_C2_values = np.array([float(C2_SAC[str(g)]) for g in article_g_list])
sac_C3_values = np.array([float(C3_SAC[str(g)]) for g in article_g_list])

#g_list = np.array([0.5, 1., 1.5, 2.])
#g_list = np.array([0.05, 0.15, 0.25, 0.4, 0.5, 0.7, 0.9, 1., 1.5, 2., 2.5, 3., 3.5, 4.])
g_list = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
#g_list = np.array([1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 4.])
#g_list = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 4.])
g_indexes = np.zeros(len(g_list), dtype=np.int32)
gs = np.concatenate((np.arange(start=0.01, stop=0.25, step=0.01),
                     np.arange(start=0.25, stop=4.05, step=0.05),
                     #np.arange(start=4.25, stop=5.25, step=0.25)
                     ))
gs = np.around(gs, decimals=2)
for i in range(len(g_list)):
    g_indexes[i] = int(np.argwhere(g_list[i] == gs)[0])
analysis_path = f'./BPS_analyzed_bounds_weak_{OPE_fix}fix'
if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)
        
article_g_indexes = np.zeros(len(article_g_list), dtype=np.int32)
for i in range(len(article_g_list)):
    article_g_indexes[i] = int(np.argwhere(article_g_list[i] == gs)[0])    


path_list = [
    #join('.', 'results_BPS', 'results_BPS_0fix_g05'),
    #join('.', 'results_BPS', 'results_BPS_0fix_g1'),
    #join('.', 'results_BPS', 'results_BPS_0fix_g15'),
    #join('.', 'results_BPS', 'results_BPS_0fix_g2'),
    #join('.', 'results_BPS', 'results_BPS_0fix_g005'),
    #join('.', 'results_BPS', 'results_BPS_0fix_g015'),
    #join('.', 'results_BPS', 'results_BPS_0fix_g025'),
    #join('.', 'results_BPS', 'results_BPS_0fix_g04'),
    #join('.', 'results_BPS', 'results_BPS_0fix_g05'),
    #join('.', 'results_BPS', 'results_BPS_0fix_g07'),
    #join('.', 'results_BPS', 'results_BPS_0fix_g09'),
    #join('.', 'results_BPS', 'results_BPS_0fix_g1'),
    #join('.', 'results_BPS', 'results_BPS_0fix_g15'),
    #join('.', 'results_BPS', 'results_BPS_0fix_g2'),
    #join('.', 'results_BPS', 'results_BPS_0fix_g25'),
    #join('.', 'results_BPS', 'results_BPS_0fix_g3'),
    #join('.', 'results_BPS', 'results_BPS_0fix_g35'),
    #join('.', 'results_BPS', 'results_BPS_0fix_g4'),
    join('.', 'results_BPS', 'results_BPS_1fix_g005'),
    join('.', 'results_BPS', 'results_BPS_1fix_g010'),
    join('.', 'results_BPS', 'results_BPS_1fix_g015'),
    join('.', 'results_BPS', 'results_BPS_1fix_g020'),
    join('.', 'results_BPS', 'results_BPS_1fix_g025'),
    join('.', 'results_BPS', 'results_BPS_1fix_g030'),
    join('.', 'results_BPS', 'results_BPS_1fix_g035'),
    join('.', 'results_BPS', 'results_BPS_1fix_g040'),
    join('.', 'results_BPS', 'results_BPS_1fix_g045'),
    join('.', 'results_BPS', 'results_BPS_1fix_g05'),
    join('.', 'results_BPS', 'results_BPS_1fix_g06'),
    join('.', 'results_BPS', 'results_BPS_1fix_g07'),
    join('.', 'results_BPS', 'results_BPS_1fix_g08'),
    join('.', 'results_BPS', 'results_BPS_1fix_g09'),
    join('.', 'results_BPS', 'results_BPS_1fix_g1'),
    #join('.', 'results_BPS', 'results_BPS_1fix_g125'),
    #join('.', 'results_BPS', 'results_BPS_1fix_g15'),
    #join('.', 'results_BPS', 'results_BPS_1fix_g175'),
    #join('.', 'results_BPS', 'results_BPS_1fix_g2'),
    #join('.', 'results_BPS', 'results_BPS_1fix_g225'),
    #join('.', 'results_BPS', 'results_BPS_1fix_g25'),
    #join('.', 'results_BPS', 'results_BPS_1fix_g275'),
    #join('.', 'results_BPS', 'results_BPS_1fix_g3'),
    #join('.', 'results_BPS', 'results_BPS_1fix_g325'),
    #join('.', 'results_BPS', 'results_BPS_1fix_g35'),
    #join('.', 'results_BPS', 'results_BPS_1fix_g4')
]

experiments = len(path_list)

def get_lambda_bounds(g_indexes):
    g_num = g_indexes.shape[0]
    OPE1_upper_bounds = np.zeros(g_num)
    OPE1_lower_bounds = np.zeros(g_num)
    OPE2_upper_bounds = np.zeros(g_num)
    OPE2_lower_bounds = np.zeros(g_num)
    OPE3_upper_bounds = np.zeros(g_num)
    OPE3_lower_bounds = np.zeros(g_num)
    for i, g_index in enumerate(g_indexes):
        OPE1_lower_bounds[i] = bounds_OPE1[g_index, 0]
        OPE1_upper_bounds[i] = bounds_OPE1[g_index, 1]
        OPE2_lower_bounds[i] = bounds_OPE2[g_index, 0]
        OPE2_upper_bounds[i] = bounds_OPE2[g_index, 1]
        OPE3_lower_bounds[i] = bounds_OPE3[g_index, 0]
        OPE3_upper_bounds[i] = bounds_OPE3[g_index, 1]
    
    return OPE1_lower_bounds, OPE1_upper_bounds, OPE2_lower_bounds, OPE2_upper_bounds, OPE3_lower_bounds, OPE3_upper_bounds
        

mean_OPE1 = np.zeros(experiments)
mean_OPE2 = np.zeros(experiments)
mean_OPE3 = np.zeros(experiments)
mean_OPEsum = np.zeros(experiments)
std_OPE1 = np.zeros(experiments)
std_OPE2 = np.zeros(experiments)
std_OPE3 = np.zeros(experiments)
std_OPEsum = np.zeros(experiments)
all_OPE1= np.zeros((experiments, rew_to_take))
all_OPE2 = np.zeros((experiments, rew_to_take))
all_OPE3 = np.zeros((experiments, rew_to_take))
all_OPEsum = np.zeros((experiments, rew_to_take))

for k, (g_el, path_el) in enumerate(zip(g_list, path_list)):
    

    onlyfiles = [f for f in listdir(path_el) if isfile(join(path_el, f))]
    r = re.compile('sac[0-9]+.csv')
    onlyfiles = list(filter(r.match, onlyfiles))

    n_files = len(onlyfiles)

    rewards = np.zeros(n_files)
    OPEs = np.zeros((n_files, lambda_len))
    #print(g_el)
    for i, f in enumerate(onlyfiles):
        #print(f)
        currf = open(join(path_el, f))
        csv_raw = csv.reader(currf)
        sp = list(csv_raw)
        last_index = -1
        while len(sp[last_index]) < 5:
            last_index = last_index-1
        data = sp[last_index]

        curr_rew = float(data[1])
        curr_crossing = float(data[2])
        curr_constraint1 = float(data[3])
        curr_constraint2 = float(data[4])
        curr_delta = [float(data[i]) for i in range(5, 5+delta_len)]
        curr_OPE = [float(data[i]) for i in range(5+delta_len, 5+delta_len+lambda_len)]

        rewards[i] = curr_rew
        OPEs[i] = curr_OPE
            
        currf.close()
        
    orderer = np.argsort(rewards)
    orderer = np.flip(orderer)
    OPEs_ordered = OPEs[orderer]

    vals = OPEs_ordered[:rew_to_take]

    all_OPE1[k] = vals[:,0]
    all_OPE2[k] = vals[:,1]
    all_OPE3[k] = vals[:,2]
    all_OPEsum[k] = vals[:,1]+vals[:,2]
    
    OPE_means = np.mean(vals, axis=0)
    OPE_stds = np.std(vals, axis=0)

    
    mean_OPE1[k] = np.mean(all_OPE1[k])
    mean_OPE2[k] = np.mean(all_OPE2[k])
    mean_OPE3[k] = np.mean(all_OPE3[k])
    mean_OPEsum[k] = np.mean(all_OPEsum[k])
    std_OPE1[k] = np.std(all_OPE1[k])
    std_OPE2[k] = np.std(all_OPE2[k])
    std_OPE3[k] = np.std(all_OPE3[k])
    std_OPEsum[k] = np.std(all_OPEsum[k])




    
OPE1_lower_bounds, OPE1_upper_bounds, OPE2_lower_bounds, OPE2_upper_bounds, OPE3_lower_bounds, OPE3_upper_bounds = get_lambda_bounds(g_indexes)

article_OPE1_lower_bounds, article_OPE1_upper_bounds, article_OPE2_lower_bounds, article_OPE2_upper_bounds, article_OPE3_lower_bounds, article_OPE3_upper_bounds = get_lambda_bounds(article_g_indexes)


with open(join(analysis_path, f'numerical_values_{rew_to_take}.txt'), 'w') as f:
    print(f'Best {rew_to_take} rewards', file=f)
    if OPE_fix == 0:
        print('OPE 1 means:', file=f)
        print(mean_OPE1, file=f)
        print('OPE 1 lower bounds: ', file=f)
        print(OPE1_lower_bounds, file=f)
        print('OPE 1 upper bounds: ', file=f)
        print(OPE1_upper_bounds, file=f)
        print('OPE 1 stds:', file=f)
        print(std_OPE1, file=f)
        print('OPE 1 relative error x100', file=f)
        print(100*std_OPE1/mean_OPE1, file=f)
        print('OPE 1 bound error x100', file=f)
        print(100*(OPE1_upper_bounds-OPE1_lower_bounds)/OPE1_lower_bounds, file=f)
        print('\n\n', file=f)
    print('OPE 2 means:', file=f)
    print(mean_OPE2, file=f)
    print('OPE 2 lower bounds: ', file=f)
    print(OPE2_lower_bounds, file=f)
    print('OPE 2 upper bounds: ', file=f)
    print(OPE2_upper_bounds, file=f)
    print('OPE 2 stds:', file=f)
    print(std_OPE2, file=f)
    print('OPE 2 relative error x100', file=f)
    print(100*std_OPE2/mean_OPE2, file=f)
    print('OPE 2 bound error x100', file=f)
    print(100*(OPE2_upper_bounds-OPE2_lower_bounds)/OPE2_lower_bounds, file=f)
    print('\n\n', file=f)
    print('OPE 3 means:', file=f)
    print(mean_OPE3, file=f)
    print('OPE 3 lower bounds: ', file=f)
    print(OPE3_lower_bounds, file=f)
    print('OPE 3 upper bounds: ', file=f)
    print(OPE3_upper_bounds, file=f)
    print('OPE 3 stds:', file=f)
    print(std_OPE3, file=f)
    print('OPE 3 relative error x100', file=f)
    print(100*std_OPE3/mean_OPE3, file=f)
    print('OPE 3 bound error x100', file=f)
    print(100*(OPE3_upper_bounds-OPE3_lower_bounds)/OPE3_lower_bounds, file=f)
    print('\n\n', file=f)
    print('OPE 2+3 means:', file=f)
    print(mean_OPEsum, file=f)
    print('OPE 2+3 lower bounds: ', file=f)
    print(OPE2_lower_bounds+OPE3_lower_bounds, file=f)
    print('OPE 2+3 upper bounds: ', file=f)
    print(OPE2_upper_bounds+OPE3_upper_bounds, file=f)
    print('OPE 2+3 stds:', file=f)
    print(std_OPEsum, file=f)
    print('OPE 2+3 relative error x100', file=f)
    print(100*std_OPEsum/mean_OPEsum, file=f)
    print('OPE 2+3 bound error x100', file=f)
    print(100*(OPE2_upper_bounds-OPE2_lower_bounds+OPE3_upper_bounds-OPE3_lower_bounds)/(OPE2_lower_bounds+OPE3_lower_bounds), file=f)


if OPE_fix==0:
# Initialize the figure
    f, ax = plt.subplots()
    # Show each observation with a scatterplot
    # Initialize the figure
    f, ax = plt.subplots()
    for i in range(rew_to_take):
        ax.scatter(
            x=g_list, y=all_OPE1[:,i], color='blue', alpha=.1, zorder=-1,
        )
    ax.scatter(x=g_list, y=mean_OPE1, color='blue', marker='d', label='Experimental mean', zorder=1)
    if ipopt:
        ax.scatter(x=article_g_list, y=ipopt_C1_values, color='red', label='IPOPT w/')
    if ipopt_without:
        ax.scatter(x=article_g_list, y=ipopt_without_C1_values, color='orange', label='IPOPT w/o')
    if sac:
        ax.scatter(x=article_g_list, y=sac_C1_values, color='yellow', label='SAC w/o')
    ax.fill_between(x=g_list, y1=OPE1_lower_bounds, y2=OPE1_upper_bounds, color='green', alpha=0.2, label='Theoretical bounds')
    plt.legend()
    sns.move_legend(
        ax, loc="upper right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
    )
    plt.xlabel('g')
    #plt.ylabel('Squared First OPE coefficient $C^2_1$')
    #plt.title(f'$C^2_1$ values with respect to g for best {rew_to_take} runs')
    plt.savefig(join(analysis_path, f'OPE1_{rew_to_take}.jpg'))

# Initialize the figure
f, ax = plt.subplots()
# Show each observation with a scatterplot
# Initialize the figure
f, ax = plt.subplots()
for i in range(rew_to_take):
    ax.scatter(
        x=g_list, y=all_OPE2[:,i], color='blue', alpha=.1, zorder=-1,
    )
ax.scatter(x=g_list, y=mean_OPE2, color='blue', marker='d', label='Experimental mean', zorder=1)
if ipopt:
    ax.scatter(x=article_g_list, y=ipopt_C2_values, color='red', label='IPOPT w/')
if ipopt_without:
    ax.scatter(x=article_g_list, y=ipopt_without_C2_values, color='orange', label='IPOPT w/o')
if sac:
    ax.scatter(x=article_g_list, y=sac_C2_values, color='yellow', label='SAC w/o')
ax.fill_between(x=g_list, y1=OPE2_lower_bounds, y2=OPE2_upper_bounds, color='green', alpha=0.2, label='Theoretical bounds')
#plt.legend()
#sns.move_legend(
#    ax, loc="upper right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
#)
plt.xlabel('g')
#plt.ylabel('Squared second OPE coefficient $C^2_2$')
#plt.title(f'$C^2_2$ values with respect to g for best {rew_to_take} runs')
plt.savefig(join(analysis_path, f'OPE2_{rew_to_take}.jpg'))


# Initialize the figure
f, ax = plt.subplots()
# Show each observation with a scatterplot
# Initialize the figure
f, ax = plt.subplots()
for i in range(rew_to_take):
    ax.scatter(
        x=g_list, y=all_OPE3[:,i], color='blue', alpha=.1, zorder=-1,
    )
ax.scatter(x=g_list, y=mean_OPE3, color='blue', marker='d', label='Experimental mean', zorder=1)
if ipopt:
    ax.scatter(x=article_g_list, y=ipopt_C3_values, color='red', label='IPOPT w/')
if ipopt_without:
    ax.scatter(x=article_g_list, y=ipopt_without_C3_values, color='orange', label='IPOPT w/o')
if sac:
    ax.scatter(x=article_g_list, y=sac_C3_values, color='yellow', label='SAC w/o')
ax.fill_between(x=g_list, y1=OPE3_lower_bounds, y2=OPE3_upper_bounds, color='green', alpha=0.2, label='Theoretical bounds')
#plt.legend()
#sns.move_legend(
#    ax, loc="upper right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
#)
plt.xlabel('g')
#plt.ylabel('Squared First OPE coefficient $C^2_3$')
#plt.title(f'$C^2_3$ values with respect to g for best {rew_to_take} runs')
plt.savefig(join(analysis_path, f'OPE3_{rew_to_take}.jpg'))


# Initialize the figure
f, ax = plt.subplots()
for i in range(rew_to_take):
    ax.scatter(
        x=g_list, y=all_OPEsum[:,i], color='blue', alpha=.1, zorder=-1,
    )
ax.scatter(x=g_list, y=mean_OPEsum, color='blue', marker='d', label='Experimental mean', zorder=1)
if ipopt:
    ax.scatter(x=article_g_list, y=ipopt_C2_values+ipopt_C3_values, color='red', label='IPOPT w/')
if ipopt_without:
    ax.scatter(x=article_g_list, y=ipopt_without_C2_values+ipopt_without_C3_values, color='orange', label='IPOPT w/o')
if sac:
    ax.scatter(x=article_g_list, y=sac_C2_values+sac_C3_values, color='yellow', label='SAC w/o')
ax.fill_between(x=g_list, y1=OPE2_lower_bounds+OPE3_lower_bounds, y2=OPE2_upper_bounds+OPE3_upper_bounds, color='green', alpha=0.2, label='Theoretical bounds')
#plt.legend()
#sns.move_legend(
#    ax, loc="upper right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
#)

plt.xlabel('g')
#plt.ylabel('Sum of squared OPE coefficients $C^2_2+C^2_3$')
#plt.title(f'$C^2_2+C^2_3$ values with respect to g for best {rew_to_take} runs')
plt.savefig(join(analysis_path, f'OPEsum_{rew_to_take}.jpg'))

if OPE_fix==0:
    f, ax = plt.subplots()
    plt.plot(g_list, 100*std_OPE1/mean_OPE1, label='Experimental relative error', color='blue')
    plt.plot(g_list, 100*(OPE1_upper_bounds-OPE1_lower_bounds)/OPE1_lower_bounds, label='Theoretical error', color='red')
    plt.legend()
    plt.xlabel('g')
    plt.ylabel('Relative error (x100)')
    plt.title('Experimental relative error vs. bound error in $C_1^2$ (x100)')
    plt.savefig(join(analysis_path, f'OPE1_{rew_to_take}_error.jpg'), dpi=300)

f, ax = plt.subplots()
plt.plot(g_list, 100*std_OPE2/mean_OPE2, label='Experimental relative error', color='blue')
plt.plot(g_list, 100*(OPE2_upper_bounds-OPE2_lower_bounds)/OPE2_lower_bounds, label='Theoretical error', color='red')
plt.legend()
plt.xlabel('g')
plt.ylabel('Relative error (x100)')
plt.title('Experimental relative error vs. bound error in $C_2^2$ (x100)')
plt.savefig(join(analysis_path, f'OPE2_{rew_to_take}_error.jpg'), dpi=300)

f, ax = plt.subplots()
plt.plot(g_list, 100*std_OPE3/mean_OPE3, label='Experimental relative error', color='blue')
plt.plot(g_list, 100*(OPE3_upper_bounds-OPE3_lower_bounds)/OPE3_lower_bounds, label='Theoretical error', color='red')
plt.legend()
plt.xlabel('g')
plt.ylabel('Relative error (x100)')
plt.title('Experimental relative error vs. bound error in $C_3^2$ (x100)')
plt.savefig(join(analysis_path, f'OPE3_{rew_to_take}_error.jpg'), dpi=300)

f, ax = plt.subplots()
plt.plot(g_list, 100*std_OPEsum/mean_OPEsum, label='Experimental relative error', color='blue')
plt.plot(g_list, 100*(OPE2_upper_bounds-OPE2_lower_bounds+OPE3_upper_bounds-OPE3_lower_bounds)/(OPE2_lower_bounds+OPE3_lower_bounds), label='Theoretical error', color='red')
plt.legend()
plt.xlabel('g')
plt.ylabel('Relative error (x100)')
plt.title('Experimental relative error vs. bound error in $C_2^2+C_3^2$ (x100)')
plt.savefig(join(analysis_path, f'OPEsum_{rew_to_take}_error.jpg'), dpi=300)

if OPE_fix==0:
    f,ax = plt.subplots()
    for i in range(rew_to_take):
        ax.scatter(
            x=g_list, y=(all_OPE1[:,i] - (OPE1_lower_bounds + OPE1_upper_bounds)/2)/(OPE1_upper_bounds - OPE1_lower_bounds), color='blue', alpha=.1, zorder=-1,
        )
    ax.scatter(x=g_list, y=(mean_OPE1- (OPE1_lower_bounds + OPE1_upper_bounds)/2)/(OPE1_upper_bounds - OPE1_lower_bounds), color='blue', marker='d', label='Experimental mean', zorder=1)
    if ipopt:
        ax.scatter(x=article_g_list, y=(ipopt_C1_values - (article_OPE1_lower_bounds + article_OPE1_upper_bounds)/2)/(article_OPE1_upper_bounds - article_OPE1_lower_bounds), color='red', label='IPOPT w/')
    if ipopt_without:
       ax.scatter(x=article_g_list, y=(ipopt_without_C1_values- (article_OPE1_lower_bounds + article_OPE1_upper_bounds)/2)/(article_OPE1_upper_bounds - article_OPE1_lower_bounds), color='orange', label='IPOPT w/o')
    if sac:
        ax.scatter(x=article_g_list, y=(sac_C1_values - (article_OPE1_lower_bounds + article_OPE1_upper_bounds)/2)/(article_OPE1_upper_bounds - article_OPE1_lower_bounds), color='yellow', label='SAC w/o')
    ax.fill_between(x=g_list, y1=(OPE1_lower_bounds- (OPE1_lower_bounds + OPE1_upper_bounds)/2)/(OPE1_upper_bounds - OPE1_lower_bounds), y2=(OPE1_upper_bounds - (OPE1_lower_bounds + OPE1_upper_bounds)/2)/(OPE1_upper_bounds - OPE1_lower_bounds), color='green', alpha=0.2, label='Theoretical bounds')
    plt.legend()
    sns.move_legend(
        ax, loc="upper right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
    )
    plt.xlabel('g')
    plt.ylabel('Normalized $C_1^2$ coefficient')
    plt.title(f'Normalized $C_1^2$ values with respect to g for best {rew_to_take} runs')
    plt.savefig(join(analysis_path, f'OPE1norm_{rew_to_take}.jpg'), dpi=300)

f,ax = plt.subplots()
for i in range(rew_to_take):
    ax.scatter(
        x=g_list, y=(all_OPE2[:,i] - (OPE2_lower_bounds + OPE2_upper_bounds)/2)/(OPE2_upper_bounds - OPE2_lower_bounds), color='blue', alpha=.1, zorder=-1,
    )
ax.scatter(x=g_list, y=(mean_OPE2- (OPE2_lower_bounds + OPE2_upper_bounds)/2)/(OPE2_upper_bounds - OPE2_lower_bounds), color='blue', marker='d', label='Experimental mean', zorder=1)
if ipopt:
    ax.scatter(x=article_g_list, y=(ipopt_C2_values - (article_OPE2_lower_bounds + article_OPE2_upper_bounds)/2)/(article_OPE2_upper_bounds - article_OPE2_lower_bounds), color='red', label='IPOPT w/')
if ipopt_without:
    ax.scatter(x=article_g_list, y=(ipopt_without_C2_values- (article_OPE2_lower_bounds + article_OPE2_upper_bounds)/2)/(article_OPE2_upper_bounds - article_OPE2_lower_bounds), color='orange', label='IPOPT w/o')
if sac:
    ax.scatter(x=article_g_list, y=(sac_C2_values - (article_OPE2_lower_bounds + article_OPE2_upper_bounds)/2)/(article_OPE2_upper_bounds - article_OPE2_lower_bounds), color='yellow', label='IPOPT w/o')
ax.fill_between(x=g_list, y1=(OPE2_lower_bounds- (OPE2_lower_bounds + OPE2_upper_bounds)/2)/(OPE2_upper_bounds - OPE2_lower_bounds), y2=(OPE2_upper_bounds - (OPE2_lower_bounds + OPE2_upper_bounds)/2)/(OPE2_upper_bounds - OPE2_lower_bounds), color='green', alpha=0.2, label='Theoretical bounds')
plt.legend()
sns.move_legend(
    ax, loc="upper right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
)
plt.xlabel('g')
plt.ylabel('Normalized $C_2^2$ coefficient')
plt.title(f'Normalized $C_2^2$ values with respect to g for best {rew_to_take} runs')
plt.savefig(join(analysis_path, f'OPE2norm_{rew_to_take}.jpg'), dpi=300)

f,ax = plt.subplots()
for i in range(rew_to_take):
    ax.scatter(
        x=g_list, y=(all_OPE3[:,i] - (OPE3_lower_bounds + OPE3_upper_bounds)/2)/(OPE3_upper_bounds - OPE3_lower_bounds), color='blue', alpha=.1, zorder=-1,
    )
if ipopt:   
    ax.scatter(x=article_g_list, y=(ipopt_C3_values - (article_OPE3_lower_bounds + article_OPE3_upper_bounds)/2)/(article_OPE3_upper_bounds - article_OPE3_lower_bounds), color='red', label='IPOPT w/')
if ipopt_without:
    ax.scatter(x=article_g_list, y=(ipopt_without_C3_values- (article_OPE3_lower_bounds + article_OPE3_upper_bounds)/2)/(article_OPE3_upper_bounds - article_OPE3_lower_bounds), color='orange', label='IPOPT w/o')
if sac:
    ax.scatter(x=article_g_list, y=(sac_C3_values - (article_OPE3_lower_bounds + article_OPE3_upper_bounds)/2)/(article_OPE3_upper_bounds - article_OPE3_lower_bounds), color='yellow', label='SAC w/o')
ax.scatter(x=g_list, y=(mean_OPE3- (OPE3_lower_bounds + OPE3_upper_bounds)/2)/(OPE3_upper_bounds - OPE3_lower_bounds), color='blue', marker='d', label='Experimental mean', zorder=1)
ax.fill_between(x=g_list, y1=(OPE3_lower_bounds- (OPE3_lower_bounds + OPE3_upper_bounds)/2)/(OPE3_upper_bounds - OPE3_lower_bounds), y2=(OPE3_upper_bounds - (OPE3_lower_bounds + OPE3_upper_bounds)/2)/(OPE3_upper_bounds - OPE3_lower_bounds), color='green', alpha=0.2, label='Theoretical bounds')
plt.legend()
sns.move_legend(
    ax, loc="upper right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
)
plt.xlabel('g')
plt.ylabel('Normalized $C_3^2$ coefficient')
plt.title(f'Normalized $C_3^2$ values with respect to g for best {rew_to_take} runs')
plt.savefig(join(analysis_path, f'OPE3norm_{rew_to_take}.jpg'), dpi=300)


OPEsum_lower_bounds = OPE2_lower_bounds + OPE3_lower_bounds
OPEsum_upper_bounds = OPE2_upper_bounds + OPE3_upper_bounds
article_OPEsum_lower_bounds = article_OPE2_lower_bounds + article_OPE3_lower_bounds
article_OPEsum_upper_bounds = article_OPE2_upper_bounds + article_OPE3_upper_bounds
f,ax = plt.subplots()
for i in range(rew_to_take):
    ax.scatter(
        x=g_list, y=(all_OPEsum[:,i] - (OPEsum_lower_bounds + OPEsum_upper_bounds)/2)/(OPEsum_upper_bounds - OPEsum_lower_bounds), color='blue', alpha=.1, zorder=-1,
    )
ax.scatter(x=g_list, y=(mean_OPEsum- (OPEsum_lower_bounds + OPEsum_upper_bounds)/2)/(OPEsum_upper_bounds - OPEsum_lower_bounds), color='blue', marker='d', label='Experimental mean', zorder=1)
if ipopt:
    ax.scatter(x=article_g_list, y=(ipopt_C2_values+ipopt_C3_values - (article_OPEsum_lower_bounds + article_OPEsum_upper_bounds)/2)/(article_OPEsum_upper_bounds - article_OPEsum_lower_bounds), color='red', label='IPOPT w/')
if ipopt_without:
    ax.scatter(x=article_g_list, y=(ipopt_without_C2_values+ipopt_without_C3_values - (article_OPEsum_lower_bounds + article_OPEsum_upper_bounds)/2)/(article_OPEsum_upper_bounds - article_OPEsum_lower_bounds), color='orange', label='IPOPT w/o')
if sac:
    ax.scatter(x=article_g_list, y=(sac_C2_values+sac_C3_values - (article_OPEsum_lower_bounds + article_OPEsum_upper_bounds)/2)/(article_OPEsum_upper_bounds - article_OPEsum_lower_bounds), color='yellow', label='SAC w/o')
ax.fill_between(x=g_list, y1=(OPEsum_lower_bounds- (OPEsum_lower_bounds + OPEsum_upper_bounds)/2)/(OPEsum_upper_bounds - OPEsum_lower_bounds), y2=(OPEsum_upper_bounds - (OPEsum_lower_bounds + OPEsum_upper_bounds)/2)/(OPEsum_upper_bounds - OPEsum_lower_bounds), color='green', alpha=0.2, label='Theoretical bounds')
plt.legend()
sns.move_legend(
    ax, loc="upper right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
)
plt.xlabel('g')
plt.ylabel('Normalized $C_3^2$ coefficient')
plt.title(f'Normalized $C_3^2$ values with respect to g for best {rew_to_take} runs')
plt.savefig(join(analysis_path, f'OPEsumnorm_{rew_to_take}.jpg'), dpi=300)