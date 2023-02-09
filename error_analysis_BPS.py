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

gs = np.concatenate((np.arange(start=0.01, stop=0.25, step=0.01),
                     np.arange(start=0.25, stop=4.05, step=0.05),
                     #np.arange(start=4.25, stop=5.25, step=0.25)
                     ))
gs = np.around(gs, decimals=2)
g = 0.25
g_index = np.argwhere(gs==g)[0]

g_list = [0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.]
path_list = [
    join('.', 'results_BPS_3fix_g05'),
    join('.', 'results_BPS_3fix_g1'),
    join('.', 'results_BPS_3fix_g15'),
    join('.', 'results_BPS_3fix_g2'),
    join('.', 'results_BPS_3fix_g25'),
    join('.', 'results_BPS_3fix_g3'),
    join('.', 'results_BPS_3fix_g35'),
    join('.', 'results_BPS_3fix_g4')
]
mean_OPE6 = np.zeros(8)
mean_OPE5 = np.zeros(8)
std_OPE6 = np.zeros(8)
std_OPE5 = np.zeros(8)
dist_OPE = np.zeros(8)

for k, (g_el, path_el) in enumerate(zip(g_list, path_list)):
    

    onlyfiles = [f for f in listdir(path_el) if isfile(join(path_el, f))]
    r = re.compile('sac[0-9]+.csv')
    onlyfiles = list(filter(r.match, onlyfiles))

    #onlyfiles = onlyfiles[:400]

    n_files = len(onlyfiles)
    #print(onlyfiles)
    best_reward = 0.
    delta_len = 10
    lambda_len = 10

    lambda_fix = 3

    rewards = np.zeros(n_files)
    OPEs = np.zeros((n_files, lambda_len-lambda_fix))

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
        OPEs[i] = curr_OPE[lambda_fix:]
            
        currf.close()
        
    orderer = np.argsort(rewards)

    OPEs_ordered = OPEs[orderer]

    best_rew_to_take = 10

    vals = OPEs_ordered[-best_rew_to_take:]
    OPE_means = np.mean(vals, axis=0)
    OPE_stds = np.std(vals, axis=0)
    
    dist_OPE[k] = np.abs(values_BPS.delta5[str(g_el)] - values_BPS.delta6[str(g_el)])
    mean_OPE5[k] = OPE_means[1]
    mean_OPE6[k] = OPE_means[2]
    std_OPE5[k] = OPE_stds[1]
    std_OPE6[k] = OPE_stds[2]
    

plt.figure()
plt.scatter(x=dist_OPE, y=std_OPE5/mean_OPE5, color='blue')
for i in range(8):
    plt.text(x=dist_OPE[i]+0.0005, y=std_OPE5[i]/mean_OPE5[i]+0.01, s=str(g_list[i]))
plt.ylabel('Standard deviation/mean')
plt.xlabel('Distance between delta_5 and delta_6')
plt.title(f'Percentage error w.r.t. distance best {best_rew_to_take} rewards, OPE5')
plt.savefig(f'std_analysis_BPS_OPE5_{best_rew_to_take}.jpg')

plt.figure()
plt.scatter(x=dist_OPE, y=std_OPE6/mean_OPE6, color='blue')
for i in range(8):
    plt.text(x=dist_OPE[i]+0.0005, y=std_OPE6[i]/mean_OPE6[i]+0.01, s=str(g_list[i]))
plt.ylabel('Standard deviation/mean')
plt.xlabel('Distance between delta_5 and delta_6')
plt.title(f'Percentage error w.r.t. distance best {best_rew_to_take} rewards, OPE6')
plt.savefig(f'std_analysis_BPS_OPE6_{best_rew_to_take}.jpg')