from os import listdir
from os.path import isfile, join
import csv
import numpy as np
from environment.utils import output_to_file
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import os

# Is this the sigma correlator?
sigma = False

# Values for analysis set regarding the correlator used
if sigma:
    delta_len = 16
    lambda_len = 16
    suffix = 'sigma'
    correlator = 'sigma'
    teor_rew = 10590097.0153
    delta_teor = np.array([1., 4., 8., 9., 2., 6., 10., 4., 5., 8., 6., 7., 10., 8., 9., 10.])
    lambda_teor = np.array([1./4, 1./4096, 81./1677721600, 1./1073741824,  1./64, 9./2621440, 45./30064771072, 9./40960, 1./65536,
                            25./234881024, 25./3670016, 1./1310720, 15527./3685081939968, 15527./57579405312, 1125./30064771072, 251145./20882130993152])
    operators = np.arange(start=1, stop=delta_len+1)
    spin_list = np.array([0, 0, 0, 0, 2, 2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 10])
else:
    delta_len = 11
    lambda_len = 11
    suffix = 'eps'
    correlator = 'epsilon'
    delta_teor = np.array([4., 8., 2., 6., 10., 4., 8., 6., 10., 8., 10.])
    lambda_teor = np.array([1., 1/100., 1., 1/10., 1/1260., 1/10., 1/126., 1/126., 1/1716., 1/1716., 1/24310.])
    operators = np.arange(start=1, stop=delta_len+1)
    spin_list = np.array([0, 0, 2, 2, 2, 4, 4, 6, 6, 8, 10])
    teor_rew = 2586.0985


# Get the results files
path = join('.', 'results_ising2D', f'results_{suffix}')
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

if not os.path.exists(f'analyzed_{suffix}'):
    os.makedirs(f'analyzed_{suffix}')

# Initiate the vectors
rewards = []
params = []
deltas_all = []
lambdas_all = []

# Cycle through the files
best_reward = 0.
for f in onlyfiles:
    # Open the file and get the list line that is the best reward got
    currf = open(join(path, f), "r")
    csv_raw = csv.reader(currf)
    sp = list(csv_raw)
    data = sp[-1]
    par = sp[1]
    params.append(par)
    
    # Check the line is correct
    if len(data)>10:
        # Get the values required
        curr_rew = float(data[1])
        rewards.append(curr_rew)
        curr_delta = [float(data[i]) for i in range(2, 2+delta_len)]
        deltas_all.append(curr_delta)
        curr_lambda = [float(data[i]) for i in range(2+delta_len, 2+delta_len+lambda_len)]
        lambdas_all.append(curr_lambda)
        # Update the best run
        if curr_rew > best_reward:
            best_run = float(data[0])
            best_reward = curr_rew
            deltas = curr_delta
            lambdas = curr_lambda
    currf.close()

# Print the best run results
with open(join(f'analyzed_{suffix}', 'best_summary.txt'), mode='w') as f:
    print(f'Theoretical best reward: {teor_rew}', file=f)
    print(f'Best run: {best_run}', file=f)
    print(f'Best reward: {best_reward}', file=f)
    print(f'Deltas: {deltas}', file=f)
    print(f'Delta relative errors: {np.abs(delta_teor - deltas)/delta_teor}', file=f)
    print(f'Lambdas: {lambdas}', file=f)
    print(f'Lambdas relative errors: {np.abs(lambda_teor - lambdas)/lambda_teor}', file=f)

# Order with respect to the reward and output a file containing the best values for the run in order
orderer = np.argsort(rewards)
for el in reversed(orderer):
    file_out = np.concatenate(([el], [rewards[el]], deltas_all[el], lambdas_all[el]))
    output_to_file(join(f'analyzed_{suffix}', 'analysis.csv'), output=file_out, mode='a')
    

### PLOTTING PART

# Open the just created analysis file
currf = open(join(f'analyzed_{suffix}', 'analysis.csv'), "r")
csv_raw = csv.reader(currf)
sp = list(csv_raw)

# Number of best runs to take
rew_to_take = 25
# Get the values for rewards, delta and lambda
lambdas = np.zeros((rew_to_take, lambda_len))
deltas = np.zeros((rew_to_take, delta_len))
rewards = np.zeros(rew_to_take)
for line in range(rew_to_take):
    rewards[line] = sp[line][1]
    deltas[line] = sp[line][2:2+delta_len]
    lambdas[line] = sp[line][2+delta_len:2+delta_len+lambda_len]

avg_err_deltas = np.mean(abs(deltas - np.tile(delta_teor, (rew_to_take, 1))/np.tile(delta_teor, (rew_to_take, 1))) ,axis=0)
avg_err_lambdas = np.mean(abs(lambdas - np.tile(lambda_teor, (rew_to_take, 1))/np.tile(lambda_teor, (rew_to_take, 1))) ,axis=0)

with open(join(f'analyzed_{suffix}', 'average_summary.txt'), mode='w') as f:
    print(f'Number of runs considered: {rew_to_take}', file=f)
    print(f'Average reward: {np.mean(rewards)}', file=f)
    print(f'Reward standard deviation: {np.std(rewards)}', file=f)
    print(f'Delta averages: {np.mean(deltas, axis=0)}', file=f)
    print(f'Delta standard deviations: {np.std(deltas, axis=0)}', file=f)
    print(f'Delta relative std: {np.std(deltas, axis=0)/np.mean(deltas, axis=0)}', file=f)
    print(f'Delta relative error mean: {avg_err_deltas}', file=f)
    print(f'OPE coefficient averages: {np.mean(lambdas, axis=0)}', file=f)
    print(f'OPE coefficient standard deviation: {np.std(lambdas, axis=0)}', file=f)
    print(f'OPE coefficient relative std: {np.std(lambdas, axis=0)/np.mean(lambdas, axis=0)}', file=f)
    print(f'OPE coefficient average relative error: {avg_err_lambdas}', file=f)


### Average plot
# Initialize the figure
f, ax = plt.subplots()
# Show each observation with a scatterplot
for i in range(rew_to_take):
    sns.stripplot(
        x=operators, y=deltas[i], color='blue',
        dodge=True, alpha=.25, zorder=1, legend=False
    )
# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(
    x=operators, y=np.mean(deltas, axis=0), color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)
sns.pointplot(
    x=operators, y=delta_teor, color='red',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Theoretical value'
)
plt.legend()
# Improve the legend
sns.move_legend(
    ax, loc="upper right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
)
plt.xlabel('Operator number $i$')
plt.ylabel('Scaling dimension $\Delta_i$')
plt.title(f'Scaling dimensions for {correlator} correlator on best {rew_to_take} runs')
plt.savefig(join(f'analyzed_{suffix}', f'delta_best_{rew_to_take}.jpg'), dpi=300)

### Best one plot
# Initialize the figure
f, ax = plt.subplots()
# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(
    x=operators, y=np.mean(deltas, axis=0), color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental value'
)
sns.pointplot(
    x=operators, y=delta_teor, color='red',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Theoretical value'
)
plt.legend()
# Improve the legend
sns.move_legend(
    ax, loc="lower right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
)
plt.xlabel('Operator number $i$')
plt.ylabel('Scaling dimension $\Delta_i$')
plt.title(f'Scaling dimensions for {correlator} correlator on the best run')
plt.savefig(join(f'analyzed_{suffix}', f'delta_best_run.jpg'), dpi=300)


### Average plot
# Initialize the figure
f, ax = plt.subplots()
# Show each observation with a scatterplot
for i in range(rew_to_take):
    sns.stripplot(
        x=operators, y=lambdas[i], color='blue',
        dodge=True, alpha=.25, zorder=1, legend=False
    )
# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(
    x=operators, y=np.mean(lambdas, axis=0), color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental mean'
)
sns.pointplot(
    x=operators, y=lambda_teor, color='red',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Theoretical value'
)
plt.legend()
# Improve the legend
sns.move_legend(
    ax, loc="upper right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
)
plt.xlabel('Operator number $i$')
plt.ylabel('Squared OPE coefficient $C^2_i$')
plt.title(f'Squared OPE coefficients for {correlator} correlator on best {rew_to_take} runs')
plt.savefig(join(f'analyzed_{suffix}', f'lambda_best_{rew_to_take}.jpg'), dpi=300)

### Best one plot
# Initialize the figure
f, ax = plt.subplots()
#sns.despine(bottom=True, left=True)
# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(
    x=operators, y=np.mean(lambdas, axis=0), color='blue',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Experimental value'
)
sns.pointplot(
    x=operators, y=lambda_teor, color='red',
    join=False, dodge=.8 - .8 / 3,
    markers="d", scale=.75, errorbar=None, label='Theoretical value'
)
plt.legend()
# Improve the legend
sns.move_legend(
    ax, loc="upper right", ncol=1, frameon=True, columnspacing=1, handletextpad=0
)
plt.xlabel('Operator number $i$')
plt.ylabel('Squared OPE coefficient $C^2_i$')
plt.title(f'Squared OPE coefficients for {correlator} correlator on the best run')

plt.savefig(join(f'analyzed_{suffix}', f'lambda_best_run.jpg'), dpi=300)