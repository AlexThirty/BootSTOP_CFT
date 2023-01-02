from ast import Param
import sys
from parameters_BPS import ParametersBPS_SAC
from environment.blocks_BPS import BPS_SAC
from environment.data_z_sample import ZData
import environment.utils as utils
from neural_net.sac_BPS import soft_actor_critic
import argparse
import os
import ray
import itertools
import numpy as np
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--faff_max', type=int, default=10000, help='Maximum number of steps without improving')
    parser.add_argument('--pc_max', type=int, default=10, help='Maximum number of reinitializations before reducing window')
    parser.add_argument('--window_rate', type=float, default=0.7, help='Rate of search window reduction')
    parser.add_argument('--max_window_exp', type=int, default=25, help='Maximun number of window reductions')
    parser.add_argument('--same_spin_hierarchy', type=bool, default=False, help='Whether same spin deltas should be ordered')
    parser.add_argument('--dyn_shift', type=float, default=0., help='Minimum distance between same spin deltas')
    parser.add_argument('--alpha', type=float, default=0.0005, help='Learning rate for actor network')
    parser.add_argument('--beta', type=float, default=0.0005, help='Learning rate for critic and value network')
    parser.add_argument('--reward_scale', type=float, default=0.001, help='The reward scale, also related to the entropy parameter')
    parser.add_argument('--gamma', type=float, default=0.99, help='Gamma parameter for cumulative reward')
    parser.add_argument('--tau', type=float, default=0.0005, help='Tau parameter for state-value function update')
    parser.add_argument('--layer1_size', type=int, default=256, help='Dense units for first layer')
    parser.add_argument('--layer2_size', type=int, default=256, help='Dense units for the second layer')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--max_cpus', type=int, default=1100, help='Maximum number of CPUs')
    parser.add_argument('--cpus_per_job', type=int, default=1, help='Maximum number of CPUs per job')
    parser.add_argument('--runs_per_args', type=int, default=25, help='Number of runs for each combination of parameters')
    
    args = parser.parse_args()
    
    gs = np.concatenate((np.arange(start=0.01, stop=0.25, step=0.01), np.arange(start=0.25, stop=4.05, step=0.05)))
    gs = np.around(gs, decimals=2)
    
    g = 1.
    integral_mode = 2
    
    ray.init(address='172.16.18.254:6379', _node_ip_address="172.16.18.254")
    print("Connected to Ray cluster.")
    print(f"Available nodes: {ray.nodes()}")

    assert ray.is_initialized(), "Error in initializing ray."

    @ray.remote(num_cpus=args.cpus_per_job)
    def run_exp(func,
                max_window_changes,
                window_decrease_rate,
                pc_max,
                file_name,
                file_name_steps,
                array_index,
                lower_bounds,
                search_window_sizes,
                guessing_run_list,
                environment_dim,
                search_space_dim,
                faff_max,
                starting_reward,
                x0,
                agent_config,
                verbose):
        soft_actor_critic(func=func,
                      max_window_changes=max_window_changes,
                      window_decrease_rate=window_decrease_rate,
                      pc_max=pc_max,
                      file_name=file_name,
                      file_name_steps=file_name_steps,
                      array_index=array_index,
                      lower_bounds=lower_bounds,
                      search_window_sizes=search_window_sizes,
                      guessing_run_list=guessing_run_list,
                      environment_dim=environment_dim,
                      search_space_dim=search_space_dim,
                      faff_max=faff_max,
                      starting_reward=starting_reward,
                      x0=x0,
                      agent_config=agent_config,
                      verbose=verbose)

    
    run_config = {}
    run_config['faff_max'] = args.faff_max
    run_config['pc_max'] = args.pc_max
    run_config['window_rate'] = args.window_rate
    run_config['max_window_exp'] = args.max_window_exp
    run_config['same_spin_hierarchy'] = args.same_spin_hierarchy
    run_config['dyn_shift'] = args.dyn_shift
    run_config['reward_scale'] = args.reward_scale
    
    agent_config = {}
    agent_config['alpha'] = args.alpha
    agent_config['beta'] = args.beta
    agent_config['reward_scale'] = args.reward_scale
    agent_config['gamma'] = args.gamma
    agent_config['tau'] = args.tau
    agent_config['rew_scale_schedule'] = 0
    agent_config['layer1_size'] = args.layer1_size
    agent_config['layer2_size'] = args.layer2_size
    agent_config['batch_size'] = args.batch_size
    agent_config['integral_mode'] = integral_mode
    agent_config['output_steps'] = 1
    agent_config['mean_output_k'] = 1000
    
    
    ab = [0.0001, 0.0005, 0.001, 0.005]
    scale = [0.001, 0.01, 0.1, 1., 10.]
    
    # ---Instantiating some relevant classes---
    params = ParametersBPS_SAC(config=run_config, g=g, integral_mode=integral_mode)

    grid = list(itertools.product(ab, scale))

    remaining_ids = []
    print(len(grid))
    for i in range(len(grid)):
        ab_val = grid[i][0]
        scale_val = grid[i][1]
        agent_config['w1'] = 1000.
        agent_config['w2'] = 10000.
        params.w1 = 1000.
        params.w2 = 10000.
        agent_config['alpha'] = ab_val
        agent_config['beta'] = ab_val
        agent_config['reward_scale'] = scale_val
        run_config['reward_scale'] = scale_val

        for j in range(args.runs_per_args):
            # ---Instantiating some relevant classes---
            zd = ZData()

            # ---Kill portion of the z-sample data if required---
            zd.kill_data(params.z_kill_list)

            g_index = np.argwhere(gs==g)[0]
            blocks = utils.generate_BPS_block_list(g_index=g_index)
            int1_list = utils.generate_BPS_int1_list(g_index=g_index)
            int2_list = utils.generate_BPS_int2_list(g_index=g_index)
            # ---Load the pre-generated conformal blocks for long multiplets---
            #blocks = utils.generate_block_list(max(params.spin_list), params.z_kill_list)

            # ---Instantiate the crossing_eqn class---
            cft = BPS_SAC(params, zd, blocks, int1_list, int2_list)

            # array_index is the cluster array number passed to the console. Set it to zero if it doesn't exist.
            array_index = args.runs_per_args*i+j

            # form the file_name where the code output is saved to
            file_name = os.path.join('results_BPS_grid_scale', params.filename_stem + str(array_index) + '.csv')
            file_name_steps = os.path.join('results_BPS_grid_scale', params.filename_stem + str(array_index) + '_steps.csv')
            output = [array_index, ab_val, scale_val]
            utils.output_to_file(file_name=file_name, output=output)
            # determine initial starting point in the form needed for the soft_actor_critic function
            x0 = params.global_best - params.shifts
            print(f'Starting run {array_index}')
            

            remaining_ids.append(run_exp.remote(func=cft.crossing_precalc,
                            max_window_changes=params.max_window_exp,
                            window_decrease_rate=params.window_rate,
                            pc_max=params.pc_max,
                            file_name=file_name,
                            file_name_steps=file_name_steps,
                            array_index=array_index,
                            lower_bounds=params.shifts,
                            search_window_sizes=params.guess_sizes,
                            guessing_run_list=params.guessing_run_list,
                            environment_dim=zd.env_shape,
                            search_space_dim=params.action_space_N,
                            faff_max=params.faff_max,
                            starting_reward=params.global_reward_start,
                            x0=x0,
                            agent_config=agent_config,
                            verbose=params.verbose))
            
            time.sleep(0.5)
        
    n_jobs = len(remaining_ids)
    print(f"Total jobs: {n_jobs}")
    #print(remaining_ids)

    # wait for jobs
    while remaining_ids:
        done_ids, remaining_ids = ray.wait(remaining_ids, num_returns=1)
        for result_id in done_ids:
            # There is only one return result by default.
            result = ray.get(result_id)
            n_jobs -= 1
            print(f'Job {result_id} terminated.\nJobs left: {n_jobs}')
