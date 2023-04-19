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

g_str = '1'
g_plot = '1'
OPE_fix = 3
rew_to_take = 10
postfix=''
#postfix = '_steps'
best_rewards_path = os.path.join('results_BPS', f'results_BPS_{OPE_fix}fix_g{g_str}')
mean_rewards_path = os.path.join('results_BPS', f'results_BPS_{OPE_fix}fix_g{g_str+postfix}')
analysis_path = 'BPS_analyzed_results'
if not os.path.exists(analysis_path):
    os.makedirs(analysis_path)

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
    resets_lab,
    filename
    ):

    #sns.set_style("whitegrid", {'axes.grid' : True,
    #                            'axes.edgecolor':'black'
    #                            })
    sns.set()
    fig = plt.figure(figsize=(20,10))
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
    plt.savefig(filename, dpi=300)
    
def plot_vanilla_bars(
    data_list,
    stds,
    timesteps,
    color_list,
    label_list,
    min_len,
    title,
    ylabel,
    resets_pos,
    resets_lab,
    filename,
    ):

    #sns.set_style("whitegrid", {'axes.grid' : True,
    #                            'axes.edgecolor':'black'
    #                            })
    fig = plt.figure(figsize=(20,10))
    plt.clf()
    ax = fig.gca()
    colors = color_list
    labels = label_list
    color_patch = []
    for color, label, data in zip(colors, labels, data_list):
        plt.plot(timesteps, data, color=color, label=label)
        #sns.lineplot(time=range(min_len), data=data, color=color, ci=95)
    print(min_len)
    #plt.xlim([0, min_len])
    plt.xlabel('Time step $(\\times10^6)$', fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.fill_between(x=timesteps, y1=data_list[0]-stds, y2=data_list[0]+stds, color=colors[0], alpha=0.2)
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
    plt.savefig(filename, dpi=300)
    
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
best_rewards = orderer[:rew_to_take]

filename = onlyfiles[best_rew]

print(f'Best run: {filename}')

div = filename.split('.')

filename_steps = f'{div[0]}_steps.csv'
filename_steps_coll = []
for rew in best_rewards:
    div = onlyfiles[rew].split('.')
    filename_steps_coll.append(f'{div[0]}_steps.csv')

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
    resets_lab=resets_labels,
    filename=os.path.join(analysis_path, 'best_reward_profile.png')
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
    resets_lab=resets_labels,
    filename=os.path.join(analysis_path, 'avg_reward_profile.png')
    )


### STEPS AVERAGES SECTION
rewards_coll = []
resets_positions_coll = []
timesteps_coll = []
resets_labels_coll = []
for filename_steps in filename_steps_coll:
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
    rewards_par = []
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
            rewards_par.append(rewards)
            #rewards = np.array()
        
    currf.close()
    
    rewards_coll.append(rewards)
    resets_positions_coll.append(resets_positions)
    timesteps_coll.append(timesteps)
    resets_labels_coll.append(resets_labels)

timesteps_min = np.min([len(timesteps) for timesteps in timesteps_coll])
print(timesteps_min)
timesteps_arg = np.argmin([len(timesteps) for timesteps in timesteps_coll])
#timesteps = timesteps_coll[np.argmin([len(timesteps) for timesteps in timesteps_coll])[0]]
rewards_final = [rewards[:timesteps_min] for rewards in rewards_coll]
rewards_final = np.array(rewards_final)
rewards_avg = np.mean(rewards_final, axis=0)
rewards_stds = np.std(rewards_final, axis=0)
resets_positions = resets_positions_coll[timesteps_arg]
resets_labels = resets_labels_coll[timesteps_arg]

print(len(rewards_avg))

plot_vanilla_bars(
    data_list=[rewards_avg],
    stds=rewards_stds,
    timesteps=np.arange(0, 100*timesteps_min, 100),
    color_list=['blue'],
    label_list=['Average reward'],
    min_len=10,
    title=f'Average reward every 100 steps, g={g_plot}',
    ylabel='Average reward',
    resets_pos=resets_positions,
    resets_lab=resets_labels,
    filename=os.path.join(analysis_path, 'avg_reward_profile_avg.png')
    )