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
g = 0.35
g_index = np.argwhere(gs==g)[0]
OPE_fix=1
path = join('/data/trenta', f'results_BPS_{OPE_fix}fix_g035')
prefix = 'g035'
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
r = re.compile('sac[0-9]+.csv')
onlyfiles = list(filter(r.match, onlyfiles))

#onlyfiles = onlyfiles[:400]

n_files = len(onlyfiles)
#print(onlyfiles)
best_reward = 0.
delta_len = 10
lambda_len = 10


rewards = np.zeros(n_files)
OPEs = np.zeros((n_files, lambda_len-OPE_fix))

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
    OPEs[i] = curr_OPE[OPE_fix:]
        
    currf.close()
    
orderer = np.argsort(rewards)
orderer = np.flip(orderer)
OPEs_ordered = OPEs[orderer]

best_rew_to_take = 10

analysis_path = f'./BPS_analysis/{prefix}'
if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)
vals = OPEs_ordered[:best_rew_to_take]
OPE_means = np.mean(vals, axis=0)
OPE_stds = np.std(vals, axis=0)
with open(join(analysis_path, f'{prefix}_OPE{OPE_fix+1}_10_{best_rew_to_take}.txt'), 'w') as f:
    print(f'Best {best_rew_to_take} rewards', file=f)
    print('OPE means:', file=f)
    print(OPE_means, file=f)
    print('OPE stds:', file=f)
    print(OPE_stds, file=f)
    print('std relative to mean', file=f)
    print(100*OPE_stds/OPE_means, file=f)

plt.figure()
for i in range(OPE_fix+1, lambda_len+1):
    plt.scatter(np.ones(best_rew_to_take)*i, vals[:, i-OPE_fix-1], color='blue')
plt.plot(range(OPE_fix+1, lambda_len+1), OPE_means, color='orange')
plt.errorbar(range(OPE_fix+1, lambda_len+1), OPE_means, yerr=OPE_stds, color='orange')
plt.xlabel('index')
plt.ylabel('OPE[index]')
plt.title(f'OPE coefficients for best {best_rew_to_take} tries')
plt.savefig(join(analysis_path, f'{prefix}_OPE{OPE_fix+1}_10_{best_rew_to_take}.jpg'))