import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import os
from os import listdir
from os.path import isfile, join
import csv
import re

integral_mode = 2
data_len = 10

g_str = '2'
g_plot = '2'
OPE_fix = 3
postfix = '_steps'
best_rewards_path = os.path.join('.', f'results_BPS_{OPE_fix}fix_g{g_str}')
mean_rewards_path = os.path.join('.', f'results_BPS_{OPE_fix}fix_g{g_str+postfix}')

print(mean_rewards_path)
def plot_vanilla(
    data_list,
    timesteps,
    color_list,
    label_list,
    min_len,
    title,
    ylabel,
    resets_pos,
    resets_lab
    ):

    #sns.set_style("whitegrid", {'axes.grid' : True,
    #                            'axes.edgecolor':'black'
    #                            })
    sns.set()
    fig = plt.figure()
    plt.clf()
    ax = fig.gca()
    colors = color_list
    labels = label_list
    color_patch = []
    for color, label, data in zip(colors, labels, data_list):
        sns.lineplot(x=timesteps, y=data, color=color, errorbar=('ci', 95))
        #sns.lineplot(time=range(min_len), data=data, color=color, ci=95)
        color_patch.append(mpatches.Patch(color=color, label=label))
    print(min_len)
    #plt.xlim([0, min_len])
    plt.xlabel('Time step $(\\times10^6)$', fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    lgd=plt.legend(
        frameon=True, fancybox=True, \
        prop={'weight':'bold', 'size':14}, handles=color_patch, loc="best")
    plt.title(title, fontsize=14)
    #ax = plt.gca()
    #ax.set_xticks([10, 20, 30, 40, 50])
    #ax.set_xticklabels([0.5, 1, 1.5, 2.5, 3.0])
    for pos, lab in zip(resets_pos, resets_lab):
        if lab == 1:
            color='red'
        else:
            color='green'
        plt.axvline(x=pos, ymin=0, ymax=np.max(data), color=color, alpha=0.2)
    plt.setp(ax.get_xticklabels(), fontsize=16)
    plt.setp(ax.get_yticklabels(), fontsize=16)
    sns.despine()
    plt.tight_layout()
    plt.show()
    
onlyfiles = [f for f in listdir(best_rewards_path) if isfile(join(best_rewards_path, f))]
r = re.compile('sac[0-9]+.csv')
onlyfiles = list(filter(r.match, onlyfiles))
onlyfiles.sort()

n_files = len(onlyfiles)

best_reward = 0.
delta_len = 10
lambda_len = 10


rewards = np.zeros(n_files)
OPEs = np.zeros((n_files, lambda_len-OPE_fix))

for i, f in enumerate(onlyfiles):
    currf = open(join(best_rewards_path, f))
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

best_rew = orderer[0]

filename = onlyfiles[best_rew]

print(f'Best run: {filename}')

div = filename.split('.')

filename_steps = f'{div[0]}_steps.csv'

print(f'Steps filename: {filename_steps}')

# Get the data from the best rewards file
currf = open(join(best_rewards_path, filename))
csv_raw = csv.reader(currf)
sp = list(csv_raw)

timesteps = np.zeros(1)
rewards = np.zeros(1)
crossings = np.zeros(1)
constraints1 = np.zeros(1)
constraints2 = np.zeros(1)
delta_data = np.zeros(data_len)
OPE_data = np.zeros(data_len)

resets_positions = []
resets_labels = []

cumulative_timestep = 0
last_cumulative_timestep = 0
rewards_in_block = 0

for i, data in enumerate(sp[1:]):
    if len(data) > 5:
        rewards_in_block += 1
        rewards = np.concatenate((rewards, [float(data[1])]))
        crossings = np.concatenate((crossings, [float(data[2])]))
        constraints1 = np.concatenate((constraints1, [float(data[3])]))
        constraints2 = np.concatenate((constraints1, [float(data[4])]))
        delta_data = np.vstack((delta_data, [float(data[j]) for j in range(3+integral_mode, 3+integral_mode+data_len)]))
        OPE_data = np.vstack((OPE_data, [float(data[j]) for j in range(3+integral_mode+data_len, 3+integral_mode+2*data_len)]))
        
    if len(data) == 3:
        cumulative_timestep = cumulative_timestep + int(data[2])
        if rewards_in_block > 0:
            ts = np.arange(start=last_cumulative_timestep, stop=cumulative_timestep, step=float(data[2])/rewards_in_block)
            timesteps = np.concatenate((timesteps, ts))
        
        resets_positions.append(last_cumulative_timestep)
        if int(data[1]) == 0:
            resets_labels.append(1)
        else:
            resets_labels.append(0)
        last_cumulative_timestep = cumulative_timestep
        rewards_in_block = 0
    
currf.close()

plot_vanilla(
    data_list=[rewards],
    timesteps=timesteps,
    color_list=['blue'],
    label_list=['Reward'],
    min_len=10,
    title=f'Best reward profile, g={g_plot}',
    ylabel='Reward',
    resets_pos=resets_positions,
    resets_lab=resets_labels
    )


### STEPS AVERAGES SECTION
currf = open(join(mean_rewards_path, filename_steps))
csv_raw = csv.reader(currf)
sp = list(csv_raw)

timesteps = np.zeros(1)
rewards = np.zeros(1)
crossings = np.zeros(1)
constraints1 = np.zeros(1)
constraints2 = np.zeros(1)
delta_data = np.zeros(data_len)
OPE_data = np.zeros(data_len)

resets_positions = []
resets_labels = []

cumulative_timestep = 0
last_cumulative_timestep = 0

for i, data in enumerate(sp):
    if len(data) > 5:
        rewards_in_block += 1
        rewards = np.concatenate((rewards, [float(data[1])]))
        crossings = np.concatenate((crossings, [float(data[2])]))
        constraints1 = np.concatenate((constraints1, [float(data[3])]))
        constraints2 = np.concatenate((constraints1, [float(data[4])]))
        delta_data = np.vstack((delta_data, [float(data[j]) for j in range(3+integral_mode, 3+integral_mode+data_len)]))
        OPE_data = np.vstack((OPE_data, [float(data[j]) for j in range(3+integral_mode+data_len, 3+integral_mode+2*data_len)]))
        timesteps = np.concatenate((timesteps, [cumulative_timestep + float(data[0])]))
        
    if len(data) == 3:
        cumulative_timestep = cumulative_timestep + int(data[2])
        resets_positions.append(last_cumulative_timestep)
        if int(data[1]) == 0:
            resets_labels.append(1)
        else:
            resets_labels.append(0)
        last_cumulative_timestep = cumulative_timestep
    
currf.close()


plot_vanilla(
    data_list=[rewards],
    timesteps=timesteps,
    color_list=['blue'],
    label_list=['Average reward'],
    min_len=10,
    title=f'Average reward every 100 steps, g={g_plot}',
    ylabel='Average reward',
    resets_pos=resets_positions,
    resets_lab=resets_labels
    )