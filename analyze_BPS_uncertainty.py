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


OPE_first = 9
OPE_second = 10
best_rew_to_take = 25
best_reward = 0.
delta_len = 10
lambda_len = 10
lambda_fix = 3

g_list = [1., 1.5, 2., 2.5, 3., 3.5, 4.]
path_list = [
    #join('.', 'results_BPS', 'results_BPS_3fix_g05'),
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



mean_OPE_first = np.zeros(experiments)
mean_OPE_second = np.zeros(experiments)
mean_OPE_sum = np.zeros(experiments)
std_OPE_first = np.zeros(experiments)
std_OPE_second = np.zeros(experiments)
std_OPE_sum = np.zeros(experiments)
dist_OPE = np.zeros(experiments)


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

    OPEs_ordered = OPEs[orderer]

    vals = OPEs_ordered[-best_rew_to_take:]
    vals_sum = vals[:, OPE_first-1] + vals[:, OPE_second-1]
    OPE_means = np.mean(vals, axis=0)
    OPE_stds = np.std(vals, axis=0)
    
    teor_deltas = get_teor_deltas(g_el)
    
    dist_OPE[k] = np.abs(teor_deltas[OPE_first-1] - teor_deltas[OPE_second-1])
    mean_OPE_first[k] = OPE_means[OPE_first-1]
    mean_OPE_second[k] = OPE_means[OPE_second-1]
    mean_OPE_sum[k] = np.mean(vals_sum)
    std_OPE_first[k] = OPE_stds[OPE_first-1]
    std_OPE_second[k] = OPE_stds[OPE_second-1]
    std_OPE_sum[k] = np.std(vals_sum)
    

plt.figure()
plt.scatter(x=dist_OPE, y=std_OPE_first/mean_OPE_first, color='blue')
for i in range(len(g_list)):
    plt.text(x=dist_OPE[i]+0.0005, y=std_OPE_first[i]/mean_OPE_first[i]+0.0001, s=f'g={str(g_list[i])}')
plt.ylabel('Standard deviation/mean')
plt.xlabel(f'Distance between delta_{OPE_first} and delta_{OPE_second}')
plt.title(f'Percentage error w.r.t. distance best {best_rew_to_take} rewards, OPE{OPE_first}')
plt.savefig(join('BPS_analyzed_uncertainty', f'uncertainty_analysis_OPE{OPE_first}_on_OPE{OPE_second}_best{best_rew_to_take}.jpg'), dpi=300)

plt.figure()
plt.scatter(x=dist_OPE, y=std_OPE_second/mean_OPE_second, color='blue')
for i in range(len(g_list)):
    plt.text(x=dist_OPE[i]+0.0005, y=std_OPE_second[i]/mean_OPE_second[i]+0.01, s=f'g={str(g_list[i])}')
plt.ylabel('Standard deviation/mean')
plt.xlabel(f'Distance between delta_{OPE_first} and delta_{OPE_second}')
plt.title(f'Percentage error w.r.t. distance best {best_rew_to_take} rewards, OPE{OPE_second}')
plt.savefig(join('BPS_analyzed_uncertainty', f'uncertainty_analysis_OPE{OPE_second}_on_OPE{OPE_first}_best{best_rew_to_take}.jpg'), dpi=300)

plt.figure()
plt.scatter(x=dist_OPE, y=std_OPE_sum/mean_OPE_sum, color='blue')
for i in range(len(g_list)):
    plt.text(x=dist_OPE[i]+0.0005, y=std_OPE_sum[i]/mean_OPE_sum[i]+0.00001, s=f'g={str(g_list[i])}')
plt.ylabel('Standard deviation/mean')
plt.xlabel(f'Distance between delta_{OPE_first} and delta_{OPE_second}')
plt.title(f'Percentage error w.r.t. distance best {best_rew_to_take} rewards, OPE{OPE_first}+OPE{OPE_second}')
plt.savefig(join('BPS_analyzed_uncertainty', f'uncertainty_analysis_sum_OPE{OPE_first}_OPE{OPE_second}_best{best_rew_to_take}.jpg'), dpi=300)