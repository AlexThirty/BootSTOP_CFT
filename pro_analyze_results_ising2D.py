from os import listdir
from os.path import isfile, join
import csv
import numpy as np
from matplotlib import pyplot as plt
from environment.utils import output_to_file

path = join('.', 'pro_results_good')
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
sigma = False

if sigma:
    delta_len = 16
    lambda_len = 16
    delta_tries = 17
    suffix = 'sigma'
    lambda_teor = np.array([1./4, 1./4096, 81./1677721600, 1./1073741824,  1./64, 9./2621440, 45./30064771072, 9./40960, 1./65536,
                            25./234881024, 25./3670016, 1./1310720, 15527./3685081939968, 15527./57579405312, 1125./30064771072, 251145./20882130993152])
    labels = ['s=0, delta=1', 's=0, delta=4', 's=0, delta=8', 's=0, delta=9', 's=2, delta=2', 's=2, delta=6', 's=2, delta=10', 's=4, delta=4',
              's=4, delta=5', 's=4, delta=8', 's=6, delta=6', 's=6, delta=7', 's=6, delta=10', 's=8, delta=8', 's=8, delta=9', 's=10, delta=10']
else:
    delta_len = 11
    lambda_len = 11
    delta_tries = 12
    suffix = 'eps'
    lambda_teor = np.array([1., 1/100., 1., 1/10., 1/1260., 1/10., 1/126., 1/126., 1/1716., 1/1716., 1/24310.])
    labels = ['s=0, delta=4', 's=0, delta=8', 's=2, delta=2', 's=2, delta=6', 's=2, delta=10', 's=4, delta=4',
              's=4, delta=8', 's=6, delta=6', 's=6, delta=10', 's=8, delta=8', 's=10, delta=10']
tries_per_deltas = 50

best_reward = 0.



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
        output_to_file(file_name=join('pro_analized_'+suffix,'pro_res_deltas'+str(i)+'.csv'), output=np.concatenate(([el], [coll[el]], deltas_coll[el], lambdas_coll[el])))
    reward_means.append(np.mean(coll))
    obt_runs.append(best_run)
    obt_rewards.append(best_reward)
    obt_deltas.append(deltas)
    obt_lambdads.append(lambdas)
    lambda_err_matrix[:, i] = abs(lambdas - lambda_teor)/lambda_teor
    lambda_err_matrix_mean[:, i] = np.mean(abs(lambdas_coll - np.tile(lambda_teor, (tries_per_deltas, 1))/np.tile(lambda_teor, (tries_per_deltas, 1))) ,axis=0)
    lambda_error_mean.append(np.mean(lam_err))
    lambda_error_best.append(np.mean(abs(lambdas - lambda_teor)/lambda_teor))
    

print(lambda_err_matrix_mean)    

for i in range(delta_tries):
    print(f'Number of deltas set: {i}')
    print(f'Best run: {obt_runs[i]}')
    print(f'Best reward: {obt_rewards[i]}')
    print(f'Best deltas: {obt_deltas[i]}')
    print(f'Best lambdas: {obt_lambdads[i]}')
    print(f'Best lambda relative error: {lambda_error_best[i]}')
    print(f'Mean lambda relative error: {lambda_error_mean[i]}')
    print(f'Reward means: {reward_means[i]}')
    


plt.plot(range(delta_tries), obt_rewards)
plt.title('Best reward w.r.t. deltas fixed')
plt.xlabel('Number of deltas fixed')
plt.ylabel('Best reward')
plt.savefig(join('pro_analized_'+suffix, 'rew_for_deltas.jpg'), dpi=300)
plt.close()

plt.plot(range(delta_tries), reward_means)
plt.title('Mean reward w.r.t. deltas fixed')
plt.xlabel('Number of deltas fixed')
plt.ylabel('Mean of best rewards')
plt.savefig(join('pro_analized_'+suffix, 'rew_for_deltas_mean.jpg'), dpi=300)
plt.close()

plt.plot(range(delta_tries), lambda_error_best)
plt.title('Mean relative error w.r.t. deltas fixed (best try)')
plt.xlabel('Number of deltas fixed')
plt.ylabel('Mean of lambda relative errors')
plt.savefig(join('pro_analized_'+suffix, 'lambda_error_best.jpg'), dpi=300)
plt.close()

plt.plot(range(delta_tries), lambda_error_mean)
plt.title('Mean relative error w.r.t. deltas fixed (mean)')
plt.xlabel('Number of deltas fixed')
plt.ylabel('Mean of lambda relative errors')
plt.savefig(join('pro_analized_'+suffix, 'lambda_error_mean.jpg'), dpi=300)
plt.close()

plt.figure()
for i in range(lambda_len):
    plt.plot(range(delta_tries), lambda_err_matrix[i, :], label=labels[i])
plt.title('Relative error w.r.t. deltas fixed (best try)')
plt.xlabel('Number of deltas fixed')
plt.ylabel('Mean of lambda relative errors')
plt.yscale('log')
plt.legend(fontsize=5)
plt.savefig(join('pro_analized_'+suffix, 'lambda_error_best_sing.jpg'), dpi=300)
plt.close()

plt.figure()
for i in range(lambda_len):
    plt.plot(range(delta_tries), lambda_err_matrix_mean[i, :], label=labels[i])
plt.title('Relative error w.r.t. deltas fixed (best try)')
plt.xlabel('Number of deltas fixed')
plt.ylabel('Mean of lambda relative errors')
plt.yscale('log')
plt.legend(fontsize=5)
plt.savefig(join('pro_analized_'+suffix, 'lambda_error_mean_sing.jpg'), dpi=300)
plt.close()