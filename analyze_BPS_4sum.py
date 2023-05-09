from os import listdir
from os.path import isfile, join
import csv
import numpy as np
from matplotlib import pyplot as plt
from scipy import special as scsp
import values_BPS
import seaborn as sns
import os
import re

weak = False

def get_avg(weak: bool):
    if weak:
        gs = np.linspace(start=0, stop=0.5, num=1000)
        avg = 1/7. + gs**2*(-1159/882. + 2*np.pi**2/21) + gs**4 * (166907/9261. - 2041*np.pi**2/1323. - 8*np.pi**4/105. + 38*scsp.zeta(3)/7)
    else:
        gs = np.linspace(start=0.5, stop=4, num=1000)
        avg = 10/429. + 257525/(2290288*gs*np.pi) + 28535513/(818777960*np.pi**2*gs**2) + (-(83060873856120557/211495299037102080.)+(45*scsp.zeta(3)/352))/(np.pi**3 * gs**3)
    return gs, avg

OPE_first = 9
OPE_second = 10
best_rew_to_take = 25
best_reward = 0.
delta_len = 10
lambda_len = 10
analysis_path = 'BPS_analyzed_results_extrabound'
if weak:
    g_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    path_list = [
        join('.', 'results_BPS', 'results_BPS_1fix_g005'),
        join('.', 'results_BPS', 'results_BPS_1fix_g010'),
        join('.', 'results_BPS', 'results_BPS_1fix_g015'),
        join('.', 'results_BPS', 'results_BPS_1fix_g020'),
        join('.', 'results_BPS', 'results_BPS_1fix_g025'),
        join('.', 'results_BPS', 'results_BPS_1fix_g030'),
        join('.', 'results_BPS', 'results_BPS_1fix_g035'),
        join('.', 'results_BPS', 'results_BPS_1fix_g040'),
        join('.', 'results_BPS', 'results_BPS_1fix_g045'),
        join('.', 'results_BPS', 'results_BPS_1fix_g05')
    ]
    lambda_fix = 1
else:
    g_list = [1., 1.5, 2., 2.5, 3., 3.5, 4.]
    path_list = [
        join('.', 'results_BPS', 'results_BPS_3fix_g1'),
        join('.', 'results_BPS', 'results_BPS_3fix_g15'),
        join('.', 'results_BPS', 'results_BPS_3fix_g2'),
        join('.', 'results_BPS', 'results_BPS_3fix_g25'),
        join('.', 'results_BPS', 'results_BPS_3fix_g3'),
        join('.', 'results_BPS', 'results_BPS_3fix_g35'),
        join('.', 'results_BPS', 'results_BPS_3fix_g4')
    ]
    lambda_fix = 3

experiments = len(path_list)


def get_teor_deltas(g):
    deltas = np.zeros(10)
    deltas[0] = values_BPS.delta1[str(g)]
    deltas[1] = values_BPS.delta2[str(g)]
    deltas[2] = values_BPS.delta3[str(g)]
    deltas[3] = values_BPS.delta4[str(g)]
    deltas[4] = values_BPS.delta5[str(g)]
    deltas[5] = values_BPS.delta6[str(g)]
    deltas[6] = values_BPS.delta7[str(g)]
    deltas[7] = values_BPS.delta8[str(g)]
    deltas[8] = values_BPS.delta9[str(g)]
    deltas[9] = values_BPS.delta10[str(g)]
    return deltas


if not os.path.exists(analysis_path):
    os.makedirs(analysis_path)

mean_OPE_4 = np.zeros(experiments)
mean_OPE_5 = np.zeros(experiments)
mean_OPE_6 = np.zeros(experiments)
mean_OPE_7 = np.zeros(experiments)
mean_OPE_8 = np.zeros(experiments)
mean_OPE_9 = np.zeros(experiments)

var_OPE_4 = np.zeros(experiments)
var_OPE_5 = np.zeros(experiments)
var_OPE_6 = np.zeros(experiments)
var_OPE_7 = np.zeros(experiments)
var_OPE_8 = np.zeros(experiments)
var_OPE_9 = np.zeros(experiments)


OPE_vals = np.zeros((best_rew_to_take, experiments, lambda_len))
OPE_m = np.zeros((lambda_len, experiments))
OPE_s = np.zeros((lambda_len, experiments))
rew_vals = np.zeros((experiments, best_rew_to_take))
rew_m = np.zeros(experiments)
rew_s = np.zeros(experiments)
rew_best = np.zeros(experiments)


for k, (g_el, path_el) in enumerate(zip(g_list, path_list)):
    onlyfiles = [f for f in listdir(path_el) if isfile(join(path_el, f))]
    r = re.compile('sac[0-9]+.csv')
    onlyfiles = list(filter(r.match, onlyfiles))

    n_files = len(onlyfiles)

    rewards = np.zeros(n_files)
    OPEs = np.zeros((n_files, lambda_len))

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
        OPEs[i] = curr_OPE
            
        currf.close()
        
    orderer = np.argsort(rewards)
    rewards_ordered = rewards[orderer]

    OPEs_ordered = OPEs[orderer]

    vals = OPEs_ordered[-best_rew_to_take:]
    OPE_vals[:,k,:] = vals
    vals_sum = vals[:, OPE_first-1] + vals[:, OPE_second-1]
    OPE_means = np.mean(vals, axis=0)
    OPE_vars = np.var(vals, axis=0)
    
    teor_deltas = get_teor_deltas(g_el)
    
    mean_OPE_4[k] = OPE_means[3]
    mean_OPE_5[k] = OPE_means[4]
    mean_OPE_6[k] = OPE_means[5]
    mean_OPE_7[k] = OPE_means[6]
    mean_OPE_8[k] = OPE_means[7]
    mean_OPE_9[k] = OPE_means[8]

    var_OPE_4[k] = OPE_vars[3]
    var_OPE_5[k] = OPE_vars[4]
    var_OPE_6[k] = OPE_vars[5]
    var_OPE_7[k] = OPE_vars[6]
    var_OPE_8[k] = OPE_vars[7]
    var_OPE_9[k] = OPE_vars[8]

#delta_raw_string = 
gs, avg = get_avg(weak=weak)

if weak:
    to_plot = np.zeros(len(g_list))
    variances = np.zeros(len(g_list))
    for i in range(len(g_list)):
        to_plot[i] = mean_OPE_4[i] + mean_OPE_5[i]+ mean_OPE_6[i] + mean_OPE_7[i] + mean_OPE_8[i] + mean_OPE_9[i]
        variances[i] = var_OPE_4[i] + var_OPE_5[i]+ var_OPE_6[i] + var_OPE_7[i] + var_OPE_8[i] + var_OPE_9[i]
        variances[i] = np.sqrt(variances[i])
    plt.figure(figsize=(8,5))
    plt.plot(gs, avg, color='green', label='Expected values')
    plt.plot(g_list, to_plot, color='red', label='Experimental values')
    plt.errorbar(g_list, y=to_plot, yerr=variances, color='red')
    plt.ylabel('Sum of squared OPE coefficients $C_4^2+C_5^2+C_6^2+C_7^2+C_8^2+C_9^2$')
    plt.xlabel(f'g')
    plt.legend()
    plt.title(f'$C_4^2+C_5^2+C_6^2+C_7^2+C_8^2+C_9^2$ expectation vs. predicted (weak coupling)')
    plt.savefig(join(analysis_path, f'4sum_analysis_weak.png'), dpi=300)
else:
    to_plot = np.zeros(len(g_list))
    variances = np.zeros(len(g_list))
    for i in range(len(g_list)):
        to_plot[i] = mean_OPE_4[i] + mean_OPE_5[i]+ mean_OPE_6[i] + mean_OPE_8[i]
        variances[i] = var_OPE_4[i] + var_OPE_5[i]+ var_OPE_6[i] + var_OPE_8[i]
        variances[i] = np.sqrt(variances[i])
    plt.figure(figsize=(8,5))
    plt.plot(gs, avg, color='green', label='Expected values')
    plt.plot(g_list, to_plot, color='red', label='Experimental values')
    plt.errorbar(g_list, y=to_plot, yerr=variances, color='red')
    plt.ylabel('Sum of squared OPE coefficients $C_4^2+C_5^2+C_6^2+C_8^2$')
    plt.xlabel(f'g')
    plt.legend()
    plt.title(f'$C_4^2+C_5^2+C_6^2+C_8^2$ expectation vs. predicted (strong coupling)')
    plt.savefig(join(analysis_path, f'4sum_analysis_strong.png'), dpi=300)
