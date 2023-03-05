from os import listdir
from os.path import isfile, join
import csv
import numpy as np
from matplotlib import pyplot as plt
from environment.utils import output_to_file
import seaborn as sns
import os

# Is this the sigma correlator?
sigma = False

# Get correlator values
if sigma:
    delta_len = 16
    lambda_len = 16
    delta_tries = delta_len+1
    suffix = 'sigma'
    delta_teor = np.array([1., 4., 8., 9., 2., 6., 10., 4., 5., 8., 6., 7., 10., 8., 9., 10.])
    lambda_teor = np.array([1./4, 1./4096, 81./1677721600, 1./1073741824,  1./64, 9./2621440, 45./30064771072, 9./40960, 1./65536,
                            25./234881024, 25./3670016, 1./1310720, 15527./3685081939968, 15527./57579405312, 1125./30064771072, 251145./20882130993152])
    labels = ['s=0, delta=1', 's=0, delta=4', 's=0, delta=8', 's=0, delta=9', 's=2, delta=2', 's=2, delta=6', 's=2, delta=10', 's=4, delta=4',
              's=4, delta=5', 's=4, delta=8', 's=6, delta=6', 's=6, delta=7', 's=6, delta=10', 's=8, delta=8', 's=8, delta=9', 's=10, delta=10']
    correlator = 'sigma'
    teor_rew = 10590097.0153
    operators = np.arange(start=1, stop=delta_len+1)
    spin_list = np.array([0, 0, 0, 0, 2, 2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 10])
else:
    delta_len = 11
    lambda_len = 11
    delta_tries = delta_len+1
    suffix = 'eps'
    correlator = 'epsilon'
    delta_teor = np.array([4., 8., 2., 6., 10., 4., 8., 6., 10., 8., 10.])
    lambda_teor = np.array([1., 1/100., 1., 1/10., 1/1260., 1/10., 1/126., 1/126., 1/1716., 1/1716., 1/24310.])
    labels = ['s=0, delta=4', 's=0, delta=8', 's=2, delta=2', 's=2, delta=6', 's=2, delta=10', 's=4, delta=4',
              's=4, delta=8', 's=6, delta=6', 's=6, delta=10', 's=8, delta=8', 's=10, delta=10']
    teor_rew = 2586.0985
    operators = np.arange(start=1, stop=delta_len+1)
    spin_list = np.array([0, 0, 2, 2, 2, 4, 4, 6, 6, 8, 10])

if not os.path.exists(f'analyzed_{suffix}_fix'):
        os.makedirs(f'analyzed_{suffix}_fix')
        
for i in range(delta_tries):
    if not os.path.exists(join(f'analyzed_{suffix}_fix', f'{i}_fix')):
        os.makedirs(join(f'analyzed_{suffix}_fix', f'{i}_fix'))
    
# Number of tries for every number of delas fixed
tries_per_deltas = 50

# Get the path of results files
path = join('.', f'results_{suffix}_fix')
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

# Initialize variables and vectors
rewards = []
params = []

obt_runs = []
obt_rewards = []
obt_deltas = []
obt_lambdads = []

reward_means = []

lambda_error_best = []
lambda_error_mean = []
lambda_err_matrix = np.zeros((lambda_len, delta_tries))
lambda_err_matrix_mean = np.zeros((lambda_len, delta_tries))
for i in range(delta_tries):
    best_reward = 0.
    coll = []
    deltas_coll = []
    lambdas_coll = []
    lam_err = []
    for j in range(tries_per_deltas):
        currf = open(join(path, 'sac'+str(100*i+j)+'.csv'))
        csv_raw = csv.reader(currf)
        sp = list(csv_raw)
        data = sp[-1]
        if len(data)>10:
            curr_rew = float(data[1])
            coll.append(curr_rew)
            curr_delta = [float(data[i]) for i in range(2, 2+delta_len)]
            deltas_coll.append(curr_delta)
            curr_lambda = [float(data[i]) for i in range(2+delta_len, 2+delta_len+lambda_len)]
            lambdas_coll.append(curr_lambda)
            lam_err.append(np.mean(abs(curr_lambda-lambda_teor)/lambda_teor))
            if curr_rew > best_reward:
                best_run = float(data[0])
                best_reward = curr_rew
                deltas = curr_delta
                lambdas = curr_lambda
        currf.close()
    orderer = np.argsort(coll)
    for el in reversed(orderer):
        output_to_file(file_name=join(f'analyzed_{suffix}_fix', f'results_deltas_{i}_fix.csv'), 
                       output=np.concatenate(([el], [coll[el]], deltas_coll[el], lambdas_coll[el])))
    reward_means.append(np.mean(coll))
    obt_runs.append(best_run)
    obt_rewards.append(best_reward)
    obt_deltas.append(deltas)
    obt_lambdads.append(lambdas)
    lambda_err_matrix[:, i] = abs(lambdas - lambda_teor)/lambda_teor
    lambda_err_matrix_mean[:, i] = np.mean(abs(lambdas_coll - np.tile(lambda_teor, (tries_per_deltas, 1))/np.tile(lambda_teor, (tries_per_deltas, 1))) ,axis=0)
    lambda_error_mean.append(np.mean(lam_err))
    lambda_error_best.append(np.mean(abs(lambdas - lambda_teor)/lambda_teor))
    
# Number of best rewards to take
rew_to_take = 10

avg_rewards = np.zeros(delta_tries)
best_rewards = np.zeros(delta_tries)
best_deltas = np.zeros((delta_tries, delta_len))
best_lambdas = np.zeros((delta_tries, lambda_len))
avg_deltas = np.zeros((delta_tries, delta_len))
avg_lambdas = np.zeros((delta_tries, lambda_len))
std_deltas = np.zeros((delta_tries, delta_len))
std_lambdas = np.zeros((delta_tries, lambda_len))
rel_deltas = np.zeros((delta_tries, delta_len))
rel_lambdas = np.zeros((delta_tries, lambda_len))
best_err_deltas = np.zeros((delta_tries, delta_len))
best_err_lambdas = np.zeros((delta_tries, lambda_len))
avg_err_deltas = np.zeros((delta_tries, delta_len))
avg_err_lambdas = np.zeros((delta_tries, lambda_len))
all_deltas = []
all_lambdas = [] 


for i in range(delta_tries):
    filename = join(f'analyzed_{suffix}_fix', f'results_deltas_{i}_fix.csv')
    currf = open(filename, "r")
    csv_raw = csv.reader(currf)
    sp = list(csv_raw)

    # Get the values for rewards, delta and lambda
    best_runs = np.zeros(rew_to_take)
    lambdas = np.zeros((rew_to_take, lambda_len))
    deltas = np.zeros((rew_to_take, delta_len))
    rewards = np.zeros(rew_to_take)
    for line in range(rew_to_take):
        best_runs[line] = sp[line][0]
        rewards[line] = sp[line][1]
        deltas[line] = sp[line][2:2+delta_len]
        lambdas[line] = sp[line][2+delta_len:2+delta_len+lambda_len]
    
    
    best_rewards[i] = rewards[0]
    avg_rewards[i] = np.mean(rewards)
    
    best_deltas[i,:] = deltas[0]
    best_lambdas[i,:] = lambdas[0]
    
    best_err_deltas[i] = np.abs(best_deltas[i]-delta_teor)/delta_teor
    best_err_lambdas[i] = np.abs(best_lambdas[i]-lambda_teor)/lambda_teor
    
    avg_deltas[i,:] = np.mean(deltas, axis=0)
    avg_lambdas[i,:] = np.mean(lambdas, axis=0)
    
    std_deltas[i,:] = np.std(deltas, axis=0)
    std_lambdas[i,:] = np.std(lambdas, axis=0)
    
    rel_deltas[i] = std_deltas[i]/avg_deltas[i]
    rel_lambdas[i] = std_lambdas[i]/avg_lambdas[i]
    
    avg_err_deltas[i] = np.mean(abs(deltas - np.tile(delta_teor, (rew_to_take, 1))/np.tile(delta_teor, (rew_to_take, 1))) ,axis=0)
    avg_err_lambdas[i] = np.mean(abs(lambdas - np.tile(lambda_teor, (rew_to_take, 1))/np.tile(lambda_teor, (rew_to_take, 1))) ,axis=0)
    
    all_deltas.append(deltas)
    all_lambdas.append(lambdas)
    
    with open(join(f'analyzed_{suffix}_fix', f'results_summary_{i}_fix.txt'), mode='w') as f:
        print(f'Theoretical best reward: {teor_rew}', file=f)
        print(f'Best run: {best_runs[0]}', file=f)
        print(f'Best reward: {best_rewards[i]}', file=f)
        print(f'Deltas: {best_deltas[i]}', file=f)
        print(f'Delta relative errors: {best_err_deltas[i]}', file=f)
        print(f'Lambdas: {best_lambdas[i]}', file=f)
        print(f'Lambdas relative errors: {best_err_lambdas[i]}', file=f)
    
    with open(join(f'analyzed_{suffix}_fix', f'result_summary_{i}fix.txt'), mode='a') as f:
        print(f'Number of runs considered: {rew_to_take}', file=f)
        print(f'Average reward: {np.mean(rewards)}', file=f)
        print(f'Reward standard deviation: {np.std(rewards)}', file=f)
        print(f'Delta averages: {np.mean(deltas, axis=0)}', file=f)
        print(f'Delta standard deviations: {np.std(deltas, axis=0)}', file=f)
        print(f'Delta relative std: {np.std(deltas, axis=0)/np.mean(deltas, axis=0)}', file=f)
        print(f'Delta average relative error: {avg_err_deltas[i]}', file=f)
        print(f'OPE coefficient averages: {np.mean(lambdas, axis=0)}', file=f)
        print(f'OPE coefficient standard deviation: {np.std(lambdas, axis=0)}', file=f)
        print(f'OPE coefficient relative std: {np.std(lambdas, axis=0)/np.mean(lambdas, axis=0)}', file=f)
        print(f'OPE coefficient average relative error: {avg_err_lambdas[i]}', file=f)
        
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
    plt.xlabel('Operator number')
    plt.ylabel('Scaling dimension')
    plt.title(f'Delta values for {correlator} correlator on best {rew_to_take} runs, {i} deltas fixed')
    plt.savefig(join(f'analyzed_{suffix}_fix', f'{i}_fix', f'delta_best_{rew_to_take}.jpg'), dpi=300)
    plt.close()


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
    plt.xlabel('Operator number')
    plt.ylabel('Scaling dimension')
    plt.title(f'Delta values for {correlator} correlator on the best single run, {i} deltas fixed')
    plt.savefig(join(f'analyzed_{suffix}_fix', f'{i}_fix', f'delta_best_run.jpg'), dpi=300)
    plt.close()

    
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
    plt.xlabel('Operator number')
    plt.ylabel('OPE coefficient')
    plt.title(f'OPE coefficients for {correlator} correlator on best {rew_to_take} runs, {i} deltas fixed')
    plt.savefig(join(f'analyzed_{suffix}_fix', f'{i}_fix', f'lambda_best_{rew_to_take}.jpg'), dpi=300)
    plt.close()


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
    plt.xlabel('Operator number')
    plt.ylabel('OPE coefficient')
    plt.title(f'OPE coefficients for {correlator} correlator on the best single run, {i} deltas fixed')

    plt.savefig(join(f'analyzed_{suffix}_fix', f'{i}_fix', f'lambda_best_run.jpg'), dpi=300)
    plt.close()

    
    
plt.plot(range(delta_tries), obt_rewards)
plt.title('Best reward w.r.t. deltas fixed')
plt.xlabel('Number of deltas fixed')
plt.ylabel('Best reward')
plt.savefig(join(f'analyzed_{suffix}_fix', 'rew_for_deltas.jpg'), dpi=300)
plt.close()

plt.plot(range(delta_tries), reward_means)
plt.title('Mean reward w.r.t. deltas fixed')
plt.xlabel('Number of deltas fixed')
plt.ylabel('Mean of best rewards')
plt.savefig(join(f'analyzed_{suffix}_fix', 'rew_for_deltas_mean.jpg'), dpi=300)
plt.close()

plt.plot(range(delta_tries), lambda_error_best)
plt.title('Mean relative error w.r.t. deltas fixed (best try)')
plt.xlabel('Number of deltas fixed')
plt.ylabel('Mean of lambda relative errors')
plt.savefig(join(f'analyzed_{suffix}_fix', 'lambda_error_best.jpg'), dpi=300)
plt.close()

plt.plot(range(delta_tries), lambda_error_mean)
plt.title('Mean relative error w.r.t. deltas fixed (mean)')
plt.xlabel('Number of deltas fixed')
plt.ylabel('Mean of lambda relative errors')
plt.savefig(join(f'analyzed_{suffix}_fix', 'lambda_error_mean.jpg'), dpi=300)
plt.close()

plt.figure()
for i in range(lambda_len):
    plt.plot(range(delta_tries), lambda_err_matrix[i, :], label=labels[i])
plt.title('Relative error w.r.t. deltas fixed (best try)')
plt.xlabel('Number of deltas fixed')
plt.ylabel('Mean of lambda relative errors')
plt.yscale('log')
plt.legend(fontsize=5)
plt.savefig(join(f'analyzed_{suffix}_fix', 'lambda_error_best_sing.jpg'), dpi=300)
plt.close()

plt.figure()
for i in range(lambda_len):
    plt.plot(range(delta_tries), lambda_err_matrix_mean[i, :], label=labels[i])
plt.title('Relative error w.r.t. deltas fixed (best try)')
plt.xlabel('Number of deltas fixed')
plt.ylabel('Mean of lambda relative errors')
plt.yscale('log')
plt.legend(fontsize=5)
plt.savefig(join(f'analyzed_{suffix}_fix', 'lambda_error_mean_sing.jpg'), dpi=300)
plt.close()