from os import listdir
from os.path import isfile, join
import csv
import numpy as np
from matplotlib import pyplot as plt
import values_BPS
import seaborn as sns
import os
import re
from scipy import special as scsp


def get_avg(weak: bool):
    if weak:
        gs = np.linspace(start=0, stop=0.5, num=1000)
        avg = 1/7. + gs**2*(-1159/882. + 2*np.pi**2/21) + gs**4 * (166907/9261. - 2041*np.pi**2/1323. - 8*np.pi**4/105. + 38*scsp.zeta(3)/7)
    else:
        gs = np.linspace(start=0.5, stop=4, num=1000)
        avg = 10/429. + 257525/(2290288*gs*np.pi) + 28535513/(818777960*np.pi**2*gs**2) + (-(83060873856120557/211495299037102080.)+(45*scsp.zeta(3)/352))/(np.pi**3 * gs**3)
    return gs, avg

OPE_first = 5
OPE_second = 6
best_rew_to_take = 25
best_reward = 0.
delta_len = 10
lambda_len = 10
lambda_fix = 1
analysis_path = 'BPS_analyzed_compare'
mode = 'all'

g_list = np.array([1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4.])

path_list_1fix = [
    join('.', 'results_BPS', 'results_BPS_1fix_g1'),
    join('.', 'results_BPS', 'results_BPS_1fix_g125'),
    join('.', 'results_BPS', 'results_BPS_1fix_g15'),
    join('.', 'results_BPS', 'results_BPS_1fix_g175'),
    join('.', 'results_BPS', 'results_BPS_1fix_g2'),
    join('.', 'results_BPS', 'results_BPS_1fix_g225'),
    join('.', 'results_BPS', 'results_BPS_1fix_g25'),
    join('.', 'results_BPS', 'results_BPS_1fix_g275'),
    join('.', 'results_BPS', 'results_BPS_1fix_g3'),
    join('.', 'results_BPS', 'results_BPS_1fix_g325'),
    join('.', 'results_BPS', 'results_BPS_1fix_g35'),
    join('.', 'results_BPS', 'results_BPS_1fix_g375'),
    join('.', 'results_BPS', 'results_BPS_1fix_g4')
]
path_list_3fix = [
    join('.', 'results_BPS', 'results_BPS_3fix_g1'),
    join('.', 'results_BPS', 'results_BPS_3fix_g125'),
    join('.', 'results_BPS', 'results_BPS_3fix_g15'),
    join('.', 'results_BPS', 'results_BPS_3fix_g175'),
    join('.', 'results_BPS', 'results_BPS_3fix_g2'),
    join('.', 'results_BPS', 'results_BPS_3fix_g225'),
    join('.', 'results_BPS', 'results_BPS_3fix_g25'),
    join('.', 'results_BPS', 'results_BPS_3fix_g275'),
    join('.', 'results_BPS', 'results_BPS_3fix_g3'),
    join('.', 'results_BPS', 'results_BPS_3fix_g325'),
    join('.', 'results_BPS', 'results_BPS_3fix_g35'),
    join('.', 'results_BPS', 'results_BPS_3fix_g375'),
    join('.', 'results_BPS', 'results_BPS_3fix_g4')
]

experiments = len(path_list_3fix)


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

mean_OPE_first_1fix = np.zeros(experiments)
mean_OPE_second_1fix = np.zeros(experiments)
mean_OPE_sum_1fix = np.zeros(experiments)
std_OPE_first_1fix = np.zeros(experiments)
std_OPE_second_1fix = np.zeros(experiments)
std_OPE_sum_1fix = np.zeros(experiments)

mean_OPE_first_3fix = np.zeros(experiments)
mean_OPE_second_3fix = np.zeros(experiments)
mean_OPE_sum_3fix = np.zeros(experiments)
std_OPE_first_3fix = np.zeros(experiments)
std_OPE_second_3fix = np.zeros(experiments)
std_OPE_sum_3fix = np.zeros(experiments)

dist_OPE = np.zeros(experiments)

OPE_vals_1fix = np.zeros((lambda_len, experiments, best_rew_to_take))
OPE_means_1fix = np.zeros((lambda_len, experiments))
OPE_stds_1fix = np.zeros((lambda_len, experiments))
rew_vals_1fix = np.zeros((experiments, best_rew_to_take))
rew_means_1fix = np.zeros(experiments)
rew_stds_1fix = np.zeros(experiments)
rew_best_1fix = np.zeros(experiments)

OPE_vals_3fix = np.zeros((lambda_len, experiments, best_rew_to_take))
OPE_means_3fix = np.zeros((lambda_len, experiments))
OPE_stds_3fix = np.zeros((lambda_len, experiments))
rew_vals_3fix = np.zeros((experiments, best_rew_to_take))
rew_means_3fix = np.zeros(experiments)
rew_stds_3fix = np.zeros(experiments)
rew_best_3fix = np.zeros(experiments)


for k, (g_el, path_el) in enumerate(zip(g_list, path_list_1fix)):
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
    rew_vals_1fix[k,:] = rewards_ordered[-best_rew_to_take:]
    rew_best_1fix[k] = rewards_ordered[-1]
    rew_means_1fix[k] = np.mean(rew_vals_1fix[k,:])
    rew_stds_1fix[k] = np.std(rew_vals_1fix[k,:])

    OPEs_ordered = OPEs[orderer]

    vals = OPEs_ordered[-best_rew_to_take:]
    OPE_vals_1fix[:,k,:] = np.transpose(vals)
    vals_sum = vals[:, OPE_first-1] + vals[:, OPE_second-1]
    OPE_means = np.mean(vals, axis=0)
    OPE_means_1fix[:,k] = OPE_means
    OPE_stds = np.std(vals, axis=0)
    OPE_stds_1fix[:,k] = OPE_stds
    
    teor_deltas = get_teor_deltas(g_el)
    
    dist_OPE[k] = np.abs(teor_deltas[OPE_first-1] - teor_deltas[OPE_second-1])
    mean_OPE_first_1fix[k] = OPE_means[OPE_first-1]
    mean_OPE_second_1fix[k] = OPE_means[OPE_second-1]
    mean_OPE_sum_1fix[k] = np.mean(vals_sum)
    std_OPE_first_1fix[k] = OPE_stds[OPE_first-1]
    std_OPE_second_1fix[k] = OPE_stds[OPE_second-1]
    std_OPE_sum_1fix[k] = np.std(vals_sum)


for k, (g_el, path_el) in enumerate(zip(g_list, path_list_3fix)):
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
    rew_vals_3fix[k,:] = rewards_ordered[-best_rew_to_take:]
    rew_best_3fix[k] = rewards_ordered[-1]
    rew_means_3fix[k] = np.mean(rew_vals_3fix[k,:])
    rew_stds_3fix[k] = np.std(rew_vals_3fix[k,:])

    OPEs_ordered = OPEs[orderer]

    vals = OPEs_ordered[-best_rew_to_take:]
    OPE_vals_3fix[:,k,:] = np.transpose(vals)
    vals_sum = vals[:, OPE_first-1] + vals[:, OPE_second-1]
    OPE_means = np.mean(vals, axis=0)
    OPE_means_3fix[:,k] = OPE_means
    OPE_stds = np.std(vals, axis=0)
    OPE_stds_3fix[:,k] = OPE_stds
    
    teor_deltas = get_teor_deltas(g_el)
    
    dist_OPE[k] = np.abs(teor_deltas[OPE_first-1] - teor_deltas[OPE_second-1])
    mean_OPE_first_3fix[k] = OPE_means[OPE_first-1]
    mean_OPE_second_3fix[k] = OPE_means[OPE_second-1]
    mean_OPE_sum_3fix[k] = np.mean(vals_sum)
    std_OPE_first_3fix[k] = OPE_stds[OPE_first-1]
    std_OPE_second_3fix[k] = OPE_stds[OPE_second-1]
    std_OPE_sum_3fix[k] = np.std(vals_sum)


    
    

### Average and best reward plotting
sns.lineplot(x=g_list, y=rew_means_1fix, color='green', label='1 Coefficient as input')
sns.lineplot(x=g_list, y=rew_means_3fix, color='blue', label='3 Coefficients as input')
plt.fill_between(x=g_list, y1=rew_means_1fix-rew_stds_1fix, y2=rew_means_1fix+rew_stds_1fix, color='green', alpha=0.2)
plt.fill_between(x=g_list, y1=rew_means_3fix-rew_stds_3fix, y2=rew_means_3fix+rew_stds_3fix, color='blue', alpha=0.2)
plt.xlabel('Coupling constant g')
plt.ylabel('Reward')
plt.legend()
plt.title(f'Average of top {best_rew_to_take} rewards as a function of g')
plt.savefig(join(analysis_path, f'rewards_for_g_best{best_rew_to_take}.jpg'), dpi=300)
plt.close()


### Plot for error on first selected OPE
plt.figure(figsize=(8,5))
plt.scatter(x=dist_OPE, y=std_OPE_first_1fix/mean_OPE_first_1fix, color='green', label='1 coefficient fixed')
for i in range(len(g_list)):
    plt.text(x=dist_OPE[i]+0.0005, y=std_OPE_first_1fix[i]/mean_OPE_first_1fix[i]+0.0001, s=f'g={str(g_list[i])}')
plt.scatter(x=dist_OPE, y=std_OPE_first_3fix/mean_OPE_first_3fix, color='blue', label='3 coefficients fixed')
for i in range(len(g_list)):
    plt.text(x=dist_OPE[i]+0.0005, y=std_OPE_first_3fix[i]/mean_OPE_first_3fix[i]+0.0001, s=f'g={str(g_list[i])}')
plt.legend()
plt.ylabel('Standard deviation/mean')
plt.xlabel(f'Distance between $\Delta_{{{OPE_first}}}$ and $\Delta_{{{OPE_second}}}$')
plt.title(f'Relative uncertainty w.r.t. distance best {best_rew_to_take} rewards, $C^2_{{{OPE_first}}}$')
plt.savefig(join(analysis_path, f'uncertainty_analysis_OPE{OPE_first}_on_OPE{OPE_second}_best{best_rew_to_take}.jpg'), dpi=300)
plt.close()

### Plot for error on second selected OPE
plt.figure(figsize=(8,5))
plt.scatter(x=dist_OPE, y=std_OPE_second_1fix/mean_OPE_second_1fix, color='green', label='1 coefficient fixed')
for i in range(len(g_list)):
    plt.text(x=dist_OPE[i]+0.0005, y=std_OPE_second_1fix[i]/mean_OPE_second_1fix[i]+0.0001, s=f'g={str(g_list[i])}')
plt.scatter(x=dist_OPE, y=std_OPE_second_3fix/mean_OPE_second_3fix, color='blue', label='3 coefficients fixed')
for i in range(len(g_list)):
    plt.text(x=dist_OPE[i]+0.0005, y=std_OPE_second_3fix[i]/mean_OPE_second_3fix[i]+0.0001, s=f'g={str(g_list[i])}')
plt.legend()
plt.ylabel('Standard deviation/mean')
plt.xlabel(f'Distance between $\Delta_{{{OPE_first}}}$ and $\Delta_{{{OPE_second}}}$')
plt.title(f'Relative uncertainty w.r.t. distance best {best_rew_to_take} rewards, $C^2_{{{OPE_second}}}$')
plt.savefig(join(analysis_path, f'uncertainty_analysis_OPE{OPE_second}_on_OPE{OPE_first}_best{best_rew_to_take}.jpg'), dpi=300)
plt.close()

### Plot for error on sum of selected OPEs
plt.figure(figsize=(8,5))
plt.scatter(x=dist_OPE, y=std_OPE_sum_1fix/mean_OPE_sum_1fix, color='green', label='1 coefficient fixed')
for i in range(len(g_list)):
    plt.text(x=dist_OPE[i]+0.0005, y=std_OPE_sum_1fix[i]/mean_OPE_sum_1fix[i]+0.0001, s=f'g={str(g_list[i])}')
plt.scatter(x=dist_OPE, y=std_OPE_sum_3fix/mean_OPE_sum_3fix, color='blue', label='3 coefficients fixed')
for i in range(len(g_list)):
    plt.text(x=dist_OPE[i]+0.0005, y=std_OPE_sum_3fix[i]/mean_OPE_sum_3fix[i]+0.0001, s=f'g={str(g_list[i])}')
plt.legend()
plt.ylabel('Standard deviation/mean')
plt.xlabel(f'Distance between $\Delta_{{{OPE_first}}}$ and $\Delta_{{{OPE_second}}}$')
plt.title(f'Relative uncertainty w.r.t. distance best {best_rew_to_take} rewards, $C^2_{{{OPE_first}}}+C^2_{{{OPE_second}}}$')
plt.savefig(join(analysis_path, f'uncertainty_analysis_sum_OPE{OPE_first}_OPE{OPE_second}_best{best_rew_to_take}.jpg'), dpi=300)
plt.close()


### Plotting of OPEs as function of g
for oper in range(lambda_fix, lambda_len):
    fig, ax = plt.subplots(figsize=(8,5))
    ### Average plot
    # Initialize the figure
    ax = sns.pointplot(
        x=g_list, y=OPE_means_1fix[oper,:], color='green',
        join=False, dodge=.8 - .8 / 3,
        markers="d", scale=.75, errorbar=None, label='1 coefficient as input', ax=ax
    )
    x_coords = []
    y_coords = []
    for point_pair in ax.collections:
        for x, y in point_pair.get_offsets():
            x_coords.append(x)
            y_coords.append(y)
    #ax.errorbar(x_coords, y_coords, yerr=OPE_stds_1fix[oper,:],
    #    color='green', fmt=' ', zorder=1)
    # Show each observation with a scatterplot
    for j in range(experiments):
        sns.stripplot(
            x=j*np.ones(best_rew_to_take), y=OPE_vals_1fix[oper,j,:], color='green',
            dodge=True, alpha=.25, zorder=-1, legend=False
        )
    sns.pointplot(
        x=g_list, y=OPE_means_3fix[oper,:], color='blue',
        join=False, dodge=.8 - .8 / 3,
        markers="d", scale=.75, errorbar=None, label='3 coefficients as input', ax=ax
    )
    #ax.errorbar(x_coords, y_coords, yerr=OPE_stds_3fix[oper,:],
    #    color='blue', fmt=' ', zorder=1)
    # Show each observation with a scatterplot
    for j in range(experiments):
        sns.stripplot(
            x=j*np.ones(best_rew_to_take), y=OPE_vals_3fix[oper,j,:], color='blue',
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
    plt.xlabel('Coupling constant g')
    plt.ylabel(f'Squared OPE coefficient $C^2_{{{oper+1}}}$')
    plt.yscale('log')
    #plt.yscale('log')
    plt.title(f'{oper+1}-th squared OPE coefficient on best {best_rew_to_take} runs, {lambda_fix} coefficient(s) fixed')
    plt.savefig(join(analysis_path, f'OPE{oper+1}_analysis_best{best_rew_to_take}.jpg'), dpi=300)

    #plt.show()
    plt.close()
    
    
for oper in range(lambda_fix, lambda_len):
    sns.lineplot(x=g_list, y=OPE_stds_1fix[oper]/OPE_means_1fix[oper], color='green', label='1 Coefficient as input')
    sns.lineplot(x=g_list, y=OPE_stds_3fix[oper]/OPE_means_3fix[oper], color='blue', label='3 Coefficients as input')
    plt.xlabel('Coupling constant g')
    plt.ylabel('Relative error')
    plt.legend()
    plt.title(f'Relative error on $C^2_{{{oper+1}}}$ as a function of g')
    plt.savefig(join(analysis_path, f'errors_for_g_OPE{oper+1}_best{best_rew_to_take}.jpg'), dpi=300)
    plt.close()
    
    
gs, avg = get_avg(weak=False)

to_plot = np.zeros(len(g_list))
variances = np.zeros(len(g_list))
for i in range(len(g_list)):
    to_plot[i] = OPE_means_1fix[3, i] + OPE_means_1fix[4, i]+ OPE_means_1fix[5, i] + OPE_means_1fix[7, i]
    variances[i] = OPE_stds_1fix[3, i]**2 + OPE_stds_1fix[4, i]**2 + OPE_stds_1fix[5, i]**2 + OPE_stds_1fix[7, i]**2
    variances[i] = np.sqrt(variances[i])
plt.figure(figsize=(8,5))
plt.plot(gs, avg, color='red', label='Expected values')
plt.plot(g_list - 0.01, to_plot, color='green', label='1 coefficient as input')
plt.errorbar(g_list - 0.01, y=to_plot, yerr=variances, color='green')

for i in range(len(g_list)):
    to_plot[i] = OPE_means_3fix[3, i] + OPE_means_3fix[4, i]+ OPE_means_3fix[5, i] + OPE_means_3fix[7, i]
    variances[i] = OPE_stds_3fix[3, i]**2 + OPE_stds_3fix[4, i]**2 + OPE_stds_3fix[5, i]**2 + OPE_stds_3fix[7, i]**2
    variances[i] = np.sqrt(variances[i])
plt.plot(g_list + 0.01, to_plot, color='blue', label='3 coefficients as input')
plt.errorbar(g_list + 0.01, y=to_plot, yerr=variances, color='blue')

plt.ylabel('Sum of squared OPE coefficients $C_4^2+C_5^2+C_6^2+C_8^2$')
plt.xlabel(f'g')
plt.legend()
plt.title(f'$C_4^2+C_5^2+C_6^2+C_8^2$ expectation vs. predicted (strong coupling)')
plt.savefig(join(analysis_path, f'4sum_analysis_strong.png'), dpi=300)
