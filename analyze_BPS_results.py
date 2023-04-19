from os import listdir
from os.path import isfile, join
import csv
import numpy as np
from matplotlib import pyplot as plt
import values_BPS
import seaborn as sns
import os
import re


OPE_first = 9
OPE_second = 10
best_rew_to_take = 25
best_reward = 0.
delta_len = 10
lambda_len = 10
lambda_fix = 1
analysis_path = 'BPS_analyzed_results_all'
#g_list = [1., 1.5, 2., 2.5, 3., 3.5, 4.]
g_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 1., 1.5, 2., 2.5, 3., 3.5, 4.]
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
    join('.', 'results_BPS', 'results_BPS_3fix_g1'),
    join('.', 'results_BPS', 'results_BPS_3fix_g15'),
    join('.', 'results_BPS', 'results_BPS_3fix_g2'),
    join('.', 'results_BPS', 'results_BPS_3fix_g25'),
    join('.', 'results_BPS', 'results_BPS_3fix_g3'),
    join('.', 'results_BPS', 'results_BPS_3fix_g35'),
    join('.', 'results_BPS', 'results_BPS_3fix_g4')
]
experiments = len(path_list)


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


if not os.path.exists(analysis_path):
    os.makedirs(analysis_path)

mean_OPE_first = np.zeros(experiments)
mean_OPE_second = np.zeros(experiments)
mean_OPE_sum = np.zeros(experiments)
std_OPE_first = np.zeros(experiments)
std_OPE_second = np.zeros(experiments)
std_OPE_sum = np.zeros(experiments)
dist_OPE = np.zeros(experiments)

OPE_vals = np.zeros((best_rew_to_take, experiments, lambda_len))
OPE_m = np.zeros((lambda_len, experiments))
OPE_s = np.zeros((lambda_len, experiments))
rew_vals = np.zeros((experiments, best_rew_to_take))
rew_m = np.zeros(experiments)
rew_s = np.zeros(experiments)
rew_best = np.zeros(experiments)


for k, (g_el, path_el) in enumerate(zip(g_list, path_list)):
    onlyfiles = [f for f in listdir(path_el) if isfile(join(path_el, f))]
    r = re.compile('sac[0-9]+.csv')
    onlyfiles = list(filter(r.match, onlyfiles))

    n_files = len(onlyfiles)

    rewards = np.zeros(n_files)
    OPEs = np.zeros((n_files, lambda_len))

    for i, f in enumerate(onlyfiles):
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
    rewards_ordered = rewards[orderer]
    rew_vals[k,:] = rewards_ordered[-best_rew_to_take:]
    rew_best[k] = rewards_ordered[-1]
    rew_m[k] = np.mean(rew_vals[k,:])
    rew_s[k] = np.std(rew_vals[k,:])

    OPEs_ordered = OPEs[orderer]

    vals = OPEs_ordered[-best_rew_to_take:]
    OPE_vals[:,k,:] = vals
    vals_sum = vals[:, OPE_first-1] + vals[:, OPE_second-1]
    OPE_means = np.mean(vals, axis=0)
    OPE_m[:,k] = OPE_means
    OPE_stds = np.std(vals, axis=0)
    OPE_s[:,k] = OPE_stds
    
    teor_deltas = get_teor_deltas(g_el)
    
    dist_OPE[k] = np.abs(teor_deltas[OPE_first-1] - teor_deltas[OPE_second-1])
    mean_OPE_first[k] = OPE_means[OPE_first-1]
    mean_OPE_second[k] = OPE_means[OPE_second-1]
    mean_OPE_sum[k] = np.mean(vals_sum)
    std_OPE_first[k] = OPE_stds[OPE_first-1]
    std_OPE_second[k] = OPE_stds[OPE_second-1]
    std_OPE_sum[k] = np.std(vals_sum)

#delta_raw_string = 
    
### Average and best reward plotting
sns.lineplot(x=g_list, y=rew_best, color='green', label='Best run reward')
sns.lineplot(x=g_list, y=rew_m, color='blue', label='Average reward')
plt.fill_between(x=g_list, y1=rew_m-rew_s, y2=rew_m+rew_s, color='blue', alpha=0.2)
plt.xlabel('Coupling constant g')
plt.ylabel('Reward')
plt.title(f'Best and average of top {best_rew_to_take} rewards as a function of g')
plt.savefig(join(analysis_path, f'rewards_for_g_best{best_rew_to_take}.jpg'), dpi=300)
plt.close()

plt.figure(figsize=(8,5))
plt.scatter(x=dist_OPE, y=std_OPE_first/mean_OPE_first, color='blue')
for i in range(len(g_list)):
    plt.text(x=dist_OPE[i]+0.0005, y=std_OPE_first[i]/mean_OPE_first[i]+0.0001, s=f'g={str(g_list[i])}')
plt.ylabel('Standard deviation/mean')
plt.xlabel(f'Distance between $\Delta_{{{OPE_first}}}$ and $\Delta_{{{OPE_second}}}$')
plt.title(f'Relative uncertainty w.r.t. distance best {best_rew_to_take} rewards, $C_{{{OPE_first}}}$')
plt.savefig(join(analysis_path, f'uncertainty_analysis_OPE{OPE_first}_on_OPE{OPE_second}_best{best_rew_to_take}.jpg'), dpi=300)

plt.figure(figsize=(8,5))
plt.scatter(x=dist_OPE, y=std_OPE_second/mean_OPE_second, color='blue')
for i in range(len(g_list)):
    plt.text(x=dist_OPE[i]+0.0005, y=std_OPE_second[i]/mean_OPE_second[i]+0.01, s=f'g={str(g_list[i])}')
plt.ylabel('Standard deviation/mean')
plt.xlabel(f'Distance between $\Delta_{{{OPE_first}}}$ and $\Delta_{{{OPE_second}}}$')
plt.title(f'Relative uncertainty w.r.t. distance best {best_rew_to_take} rewards, $C_{{{OPE_first}}}$')
plt.savefig(join(analysis_path, f'uncertainty_analysis_OPE{OPE_second}_on_OPE{OPE_first}_best{best_rew_to_take}.jpg'), dpi=300)

plt.figure(figsize=(8,5))
plt.scatter(x=dist_OPE, y=std_OPE_sum/mean_OPE_sum, color='blue')
for i in range(len(g_list)):
    plt.text(x=dist_OPE[i]+0.0005, y=std_OPE_sum[i]/mean_OPE_sum[i]+0.00001, s=f'g={str(g_list[i])}')
plt.ylabel('Standard deviation/mean')
plt.xlabel(f'Distance between $\Delta_{{{OPE_first}}}$ and $\Delta_{{{OPE_second}}}$')
plt.title(f'Relative uncertainty w.r.t. distance best {best_rew_to_take} rewards, $C_{{{OPE_first}}}+C_{{{OPE_second}}}$')
plt.savefig(join(analysis_path, f'uncertainty_analysis_sum_OPE{OPE_first}_OPE{OPE_second}_best{best_rew_to_take}.jpg'), dpi=300)

for oper in range(lambda_fix, lambda_len):
    fig, ax = plt.subplots(figsize=(8,5))
    ### Average plot
    # Initialize the figure
    ax = sns.pointplot(
        x=g_list, y=OPE_m[oper,:], color='orange',
        join=False, dodge=.8 - .8 / 3,
        markers="d", scale=.75, errorbar=None, label='Experimental mean', ax=ax
    )
    x_coords = []
    y_coords = []
    for point_pair in ax.collections:
        for x, y in point_pair.get_offsets():
            x_coords.append(x)
            y_coords.append(y)
    ax.errorbar(x_coords, y_coords, yerr=OPE_s[oper,:],
        color='orange', fmt=' ', zorder=1)
    # Show each observation with a scatterplot
    for j in range(experiments):
        sns.stripplot(
            x=j*np.ones(best_rew_to_take), y=OPE_vals[:,j,oper], color='blue',
            dodge=True, alpha=.25, zorder=-1, legend=False
        )
    # Show the conditional means, aligning each pointplot in the
    # center of the strips by adjusting the width allotted to each
    # category (.8 by default) by the number of hue levels
    plt.legend()
    # Improve the legend
    sns.move_legend(
        ax, loc="upper right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
    )
    plt.xlabel('Coupling constant g')
    plt.ylabel(f'Squared OPE coefficient $C_{{{oper+1}}}$')
    #plt.yscale('log')
    plt.title(f'{oper+1}-th squared OPE coefficient on best {best_rew_to_take} runs, {lambda_fix} coefficient fixed')
    plt.savefig(join(analysis_path, f'OPE{oper+1}_analysis_best{best_rew_to_take}.jpg'), dpi=300)

    #plt.show()
    plt.close()