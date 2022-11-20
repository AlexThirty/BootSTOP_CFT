from os import listdir
from os.path import isfile, join
import csv
import numpy as np
from matplotlib import pyplot as plt
from environment.utils import output_to_file

path = join('.', 'pro_results')
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
best_reward = 0.
delta_len = 11
lambda_len = 11

delta_tries = 12
tries_per_deltas = 25
rewards = []
params = []

obt_runs = []
obt_rewards = []
obt_deltas = []
obt_lambdads = []

reward_means = []

for i in range(delta_tries):
    best_reward = 0.
    coll = []
    deltas_coll = []
    lambdas_coll = []
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
            if curr_rew > best_reward:
                best_run = float(data[0])
                best_reward = curr_rew
                deltas = curr_delta
                lambdas = curr_lambda
        currf.close()
    orderer = np.argsort(coll)
    for el in reversed(orderer):
        output_to_file(file_name=join('analized','pro_res_deltas'+str(i)+'.csv'), output=np.concatenate(([el], [coll[el]], deltas_coll[el], lambdas_coll[el])))
    reward_means.append(np.mean(coll))
    obt_runs.append(best_run)
    obt_rewards.append(best_reward)
    obt_deltas.append(deltas)
    obt_lambdads.append(lambdas)
    
for i in range(delta_tries):
    print(f'Number of deltas set: {i}')
    print(f'Best run: {obt_runs[i]}')
    print(f'Best reward: {obt_rewards[i]}')
    print(f'Best deltas: {obt_deltas[i]}')
    print(f'Best lambdas: {obt_lambdads[i]}')
    print(f'Reward means: {reward_means[i]}')
    


plt.plot(range(delta_tries), obt_rewards)
plt.title('Best reward w.r.t. deltas fixed')
plt.xlabel('Number of deltas fixed')
plt.ylabel('Best reward')
plt.savefig('rew_for_deltas.jpg')
plt.close()

plt.plot(range(delta_tries), reward_means)
plt.title('Best reward w.r.t. deltas fixed')
plt.xlabel('Number of deltas fixed')
plt.ylabel('Mean of best rewards')
plt.savefig('rew_for_deltas_mean.jpg')