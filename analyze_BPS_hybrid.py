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

path = join('/data/trenta', 'results_BPS_deltaunk_3fix')
save_path = join('.', 'BPS_hybrid_3fix_analyzed')
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
r = re.compile('sac[0-9]+.csv')
onlyfiles = list(filter(r.match, onlyfiles))
n_files = len(onlyfiles)

best_reward = 0.

delta_len = 15
lambda_len = 15

lambda_fix = 3

rewards = np.zeros(n_files)
OPEs = np.zeros((n_files, lambda_len-lambda_fix))
deltas = np.zeros((n_files, 5))

for i, f in enumerate(onlyfiles):
    currf = open(join(path, f))
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
    deltas[i] = curr_delta[10:]
        
    currf.close()
    
orderer = np.argsort(rewards)

OPEs_ordered = OPEs[orderer]
deltas_ordered = deltas[orderer]

best_rew_to_take = 25

vals = OPEs_ordered[-best_rew_to_take:]
OPE_means = np.mean(vals, axis=0)
OPE_stds = np.std(vals, axis=0)
print(OPE_means)
print(OPE_stds)

plt.figure()
for i in range(lambda_fix+1, lambda_len+1):
    plt.scatter(np.ones(best_rew_to_take)*i, vals[:, i-lambda_fix-1], color='blue')
plt.plot(range(lambda_fix+1, lambda_len+1), OPE_means, color='orange')
plt.errorbar(range(lambda_fix+1, lambda_len+1), OPE_means, yerr=OPE_stds, color='orange')
plt.xlabel('index')
plt.ylabel('OPE[index]')
plt.title(f'OPE coefficients for best {best_rew_to_take} tries')
plt.savefig(f'hybrid_OPE4_15_{best_rew_to_take}.jpg')


vals = deltas_ordered[-best_rew_to_take:]
delta_means = np.mean(vals, axis=0)
delta_stds = np.std(vals, axis=0)
print(delta_means)
print(delta_stds)

plt.figure()
for i in range(10, 15):
    plt.scatter(np.ones(best_rew_to_take)*i, vals[:, i-10], color='blue')
plt.plot(range(10, 15), delta_means, color='orange')
plt.errorbar(range(10, 15), delta_means, yerr=delta_stds, color='orange')
plt.xlabel('index')
plt.ylabel('delta[index]')
plt.title(f'delta for best {best_rew_to_take} tries')
plt.savefig(f'hybrid_delta10_15_{best_rew_to_take}.jpg')

