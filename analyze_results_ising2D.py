from os import listdir
from os.path import isfile, join
import csv
import numpy as np
from environment.utils import output_to_file
path = join('.', 'results')
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
best_reward = 0.
sigma = True

if sigma:
    delta_len = 16
    lambda_len = 16
    suffix = 'sigma'
else:
    delta_len = 11
    lambda_len = 11
    suffix = 'eps'
    
rewards = []
params = []
deltas_all = []
lambdas_all = []

for f in onlyfiles:
    currf = open(join(path, f), "r")
    csv_raw = csv.reader(currf)
    sp = list(csv_raw)
    data = sp[-1]
    par = sp[1]
    params.append(par)
    
    if len(data)>10:
        curr_rew = float(data[1])
        rewards.append(curr_rew)
        curr_delta = [float(data[i]) for i in range(2, 2+delta_len)]
        deltas_all.append(curr_delta)
        curr_lambda = [float(data[i]) for i in range(2+delta_len, 2+delta_len+lambda_len)]
        lambdas_all.append(curr_lambda)
        if curr_rew > best_reward:
            best_run = float(data[0])
            best_reward = curr_rew
            deltas = curr_delta
            lambdas = curr_lambda
    currf.close()
    
print(f'Best run: {best_run}')
print(f'Best reward: {best_reward}')
print(f'Deltas: {deltas}')
print(f'Lambdas: {lambdas}')

orderer = np.argsort(rewards)
for el in reversed(orderer):
    file_out = np.concatenate(([el], [rewards[el]], deltas_all[el], lambdas_all[el]))
    output_to_file(join('analyzed_'+suffix, 'res_analyze.csv'), output=file_out)