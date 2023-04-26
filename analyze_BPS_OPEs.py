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
import seaborn as sns
import pandas as pd

rew_to_take = 25
gs = np.concatenate((np.arange(start=0.01, stop=0.25, step=0.01),
                     np.arange(start=0.25, stop=4.05, step=0.05),
                     #np.arange(start=4.25, stop=5.25, step=0.25)
                     ))
gs = np.around(gs, decimals=2)
g = 0.5
g_index = np.argwhere(gs==g)[0]
OPE_fix = 3
path = join('.', 'results_BPS', f'results_BPS_{OPE_fix}fix_g4')
prefix = 'g4'
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


analysis_path = f'./BPS_analyzed/{prefix}'
if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)
vals = OPEs_ordered[:rew_to_take]
OPE_means = np.mean(vals, axis=0)
OPE_stds = np.std(vals, axis=0)
with open(join(analysis_path, f'{prefix}_OPE{OPE_fix+1}_10_{rew_to_take}.txt'), 'w') as f:
    print(f'Best {rew_to_take} rewards', file=f)
    print('OPE means:', file=f)
    print(OPE_means, file=f)
    print('OPE stds:', file=f)
    print(OPE_stds, file=f)
    print('std relative to mean', file=f)
    print(100*OPE_stds/OPE_means, file=f)


operators = np.arange(start=OPE_fix+1, stop=lambda_len+1)
fig, ax = plt.subplots()
### Average plot
# Initialize the figure
ax = sns.pointplot(
    x=operators, y=OPE_means, color='orange',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean', ax=ax
)
x_coords = []
y_coords = []
for point_pair in ax.collections:
    for x, y in point_pair.get_offsets():
        x_coords.append(x)
        y_coords.append(y)
ax.errorbar(x_coords, y_coords, yerr=OPE_stds,
    color='orange', fmt=' ', zorder=1)
# Show each observation with a scatterplot
for j in range(rew_to_take):
    sns.stripplot(
        x=operators, y=OPEs_ordered[j], color='blue',
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
plt.xlabel('Operator number $i$')
plt.ylabel('Squared OPE coefficient $C^2_i$')
#plt.yscale('log')
plt.title(f'Squared OPE coefficients on best {rew_to_take} runs, {OPE_fix} coefficients fixed, g={g}')
plt.savefig(join(analysis_path, f'{prefix}_OPE{OPE_fix+1}_10_{rew_to_take}.jpg'), dpi=300)

#plt.show()
plt.close()