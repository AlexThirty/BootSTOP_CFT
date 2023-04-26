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

best_reward = 0.
delta_len = 10
lambda_len = 10
lambda_fix = 1
rew_to_take = 25


g_list = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
g_indexes = np.zeros(len(g_list), dtype=np.int32)
gs = np.concatenate((np.arange(start=0.01, stop=0.25, step=0.01),
                     np.arange(start=0.25, stop=4.05, step=0.05),
                     #np.arange(start=4.25, stop=5.25, step=0.25)
                     ))
gs = np.around(gs, decimals=2)
for i in range(len(g_list)):
    g_indexes[i] = int(np.argwhere(g_list[i] == gs)[0])
analysis_path = f'./BPS_analyzed_bounds/'
if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)


path_list = [
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
    #join('.', 'results_BPS', 'results_BPS_1fix_g15'),
    #join('.', 'results_BPS', 'results_BPS_1fix_g2'),
    #join('.', 'results_BPS', 'results_BPS_1fix_g25'),
    #join('.', 'results_BPS', 'results_BPS_1fix_g3'),
    #join('.', 'results_BPS', 'results_BPS_1fix_g35'),
    #join('.', 'results_BPS', 'results_BPS_1fix_g4')
]
experiments = len(path_list)

def get_lambda_bounds(g_indexes):
    g_num = g_indexes.shape[0]
    OPE2_upper_bounds = np.zeros(g_num)
    OPE2_lower_bounds = np.zeros(g_num)
    OPE3_upper_bounds = np.zeros(g_num)
    OPE3_lower_bounds = np.zeros(g_num)
    for i, g_index in enumerate(g_indexes):
        OPE2_lower_bounds[i] = bounds_OPE2[g_index, 0]
        OPE2_upper_bounds[i] = bounds_OPE2[g_index, 1]
        OPE3_lower_bounds[i] = bounds_OPE3[g_index, 0]
        OPE3_upper_bounds[i] = bounds_OPE3[g_index, 1]
    
    return OPE2_lower_bounds, OPE2_upper_bounds, OPE3_lower_bounds, OPE3_upper_bounds
        

mean_OPE_first = np.zeros(experiments)
mean_OPE_second = np.zeros(experiments)
std_OPE_first = np.zeros(experiments)
std_OPE_second = np.zeros(experiments)
all_OPE_first = np.zeros((experiments, rew_to_take))
all_OPE_second = np.zeros((experiments, rew_to_take))
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

    all_OPE_first[k] = vals[:,1]
    all_OPE_second[k] = vals[:,2]
    
    OPE_means = np.mean(vals, axis=0)
    OPE_stds = np.std(vals, axis=0)

    
    mean_OPE_first[k] = OPE_means[1]
    mean_OPE_second[k] = OPE_means[2]
    std_OPE_first[k] = OPE_stds[1]
    std_OPE_second[k] = OPE_stds[2]




    
OPE2_lower_bounds, OPE2_upper_bounds, OPE3_lower_bounds, OPE3_upper_bounds = get_lambda_bounds(g_indexes)

with open(join(analysis_path, f'numerical_values_{rew_to_take}.txt'), 'w') as f:
    print(f'Best {rew_to_take} rewards', file=f)
    print('OPE 2 means:', file=f)
    print(mean_OPE_first, file=f)
    print('OPE 2 stds:', file=f)
    print(std_OPE_first, file=f)
    print('OPE 3 means:', file=f)
    print(mean_OPE_second, file=f)
    print('OPE 3 stds:', file=f)
    print(std_OPE_second, file=f)
    print('std relative to mean', file=f)
    print(100*OPE_stds/OPE_means, file=f)
    print('OPE 2 lower bounds: ', file=f)
    print(OPE2_lower_bounds, file=f)
    print('OPE 2 upper bounds: ', file=f)
    print(OPE2_upper_bounds, file=f)
    print('OPE 3 lower bounds: ', file=f)
    print(OPE3_lower_bounds, file=f)
    print('OPE 3 upper bounds: ', file=f)
    print(OPE3_upper_bounds, file=f)



# Initialize the figure
f, ax = plt.subplots()
# Show each observation with a scatterplot
# Initialize the figure
f, ax = plt.subplots()
for i in range(rew_to_take):
    ax.scatter(
        x=g_list, y=all_OPE_first[:,i], color='blue', alpha=.1, zorder=-1,
    )
ax.scatter(x=g_list, y=mean_OPE_first, color='blue', marker='d', label='Experimental mean', zorder=1)
ax.fill_between(x=g_list, y1=OPE2_lower_bounds, y2=OPE2_upper_bounds, color='green', alpha=0.2, label='Theoretical bounds')
# Show each observation with a scatterplot
# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels


plt.legend()
# Improve the legend
sns.move_legend(
    ax, loc="upper right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
)


plt.xlabel('g')
plt.ylabel('Squared second OPE coefficient $C^2_2$')
plt.title(f'$C^2_2$ values with respect to g for best {rew_to_take} runs')
plt.savefig(join(analysis_path, f'OPE2_{rew_to_take}.jpg'), dpi=300)

# Initialize the figure
f, ax = plt.subplots()
# Show each observation with a scatterplot
# Initialize the figure
f, ax = plt.subplots()
for i in range(rew_to_take):
    ax.scatter(
        x=g_list, y=all_OPE_second[:,i], color='blue', alpha=.1, zorder=-1,
    )
ax.scatter(x=g_list, y=mean_OPE_second, color='blue', marker='d', label='Experimental mean', zorder=1)
ax.fill_between(x=g_list, y1=OPE3_lower_bounds, y2=OPE3_upper_bounds, color='green', alpha=0.2, label='Theoretical bounds')
# Show each observation with a scatterplot
# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels


plt.legend()
# Improve the legend
sns.move_legend(
    ax, loc="upper right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
)


plt.xlabel('g')
plt.ylabel('Squared third OPE coefficient $C^2_3$')
plt.title(f'$C^2_3$ values with respect to g for best {rew_to_take} runs')
plt.savefig(join(analysis_path, f'OPE3_{rew_to_take}.jpg'), dpi=300)


# Initialize the figure
f, ax = plt.subplots()
# Show each observation with a scatterplot
# Initialize the figure
f, ax = plt.subplots()
for i in range(rew_to_take):
    ax.scatter(
        x=g_list, y=all_OPE_second[:,i]+all_OPE_first[:,i], color='blue', alpha=.1, zorder=-1,
    )
ax.scatter(x=g_list, y=mean_OPE_second+mean_OPE_first, color='blue', marker='d', label='Experimental mean', zorder=1)
ax.fill_between(x=g_list, y1=OPE3_lower_bounds+OPE2_lower_bounds, y2=OPE3_upper_bounds+OPE2_upper_bounds, color='green', alpha=0.2, label='Theoretical bounds')
# Show each observation with a scatterplot
# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels


plt.legend()
# Improve the legend
sns.move_legend(
    ax, loc="upper right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
)


plt.xlabel('g')
plt.ylabel('Sum of squared OPE coefficients $C^2_2+C^2_3$')
plt.title(f'$C^2_2+C^2_3$ values with respect to g for best {rew_to_take} runs')
plt.savefig(join(analysis_path, f'OPEsum_{rew_to_take}.jpg'), dpi=300)