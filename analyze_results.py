from os import listdir
from os.path import isfile, join
import csv
import numpy as np
path = join('.', 'results')
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
best_reward = 0.
delta_len = 11
lambda_len = 11
tries_per_param = 500
rewards = []
params = []

for f in onlyfiles:
    currf = open(join(path, f), "r")
    csv_raw = csv.reader(currf)
    sp = list(csv_raw)
    data = sp[-1]
    par = sp[1]
    rewards.append(float(data[1]))
    params.append(par)
    
    if len(data)>10:
        curr_rew = float(data[1])
        curr_delta = [float(data[i]) for i in range(2, 2+delta_len)]
        curr_lambda = [float(data[i]) for i in range(2+delta_len, 2+delta_len+lambda_len)]
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