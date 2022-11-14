from ast import Param
import sys
from parameters_BPS import ParametersBPS_SAC
from environment.blocks_BPS import BPS_SAC
from environment.data_z_sample import ZData
import environment.utils as utils
from neural_net.sac import soft_actor_critic
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--faff_max', type=int, default=500, help='Maximum number of steps without improving')
    parser.add_argument('--pc_max', type=int, default=40, help='Maximum number of reinitializations before reducing window')
    parser.add_argument('--window_rate', type=float, default=0.7, help='Rate of search window reduction')
    parser.add_argument('--max_window_exp', type=int, default=10, help='Maximun number of window reductions')
    parser.add_argument('--same_spin_hierarchy', type=bool, default=True, help='Whether same spin deltas should be ordered')
    parser.add_argument('--dyn_shift', type=float, default=0.5, help='Minimum distance between same spin deltas')
    parser.add_argument('--alpha', type=float, default=0.0005, help='Learning rate for actor network')
    parser.add_argument('--beta', type=float, default=0.0005, help='Learning rate for critic and value network')
    parser.add_argument('--reward_scale', type=float, default=0.001, help='The reward scale, also related to the entropy parameter')
    parser.add_argument('--gamma', type=float, default=0.99, help='Gamma parameter for cumulative reward')
    parser.add_argument('--tau', type=float, default=0.0005, help='Tau parameter for state-value function update')
    parser.add_argument('--layer1_size', type=int, default=256, help='Dense units for first layer')
    parser.add_argument('--layer2_size', type=int, default=256, help='Dense units for the second layer')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--array_index', type=int, default=0, help='Index for the try in case of multitry')
    args = parser.parse_args()
    
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
    agent_config['layer1_size'] = args.layer1_size
    agent_config['layer2_size'] = args.layer2_size
    agent_config['batch_size'] = args.batch_size
    
    # ---Instantiating some relevant classes---
    params = ParametersBPS_SAC(run_config)
    zd = ZData()

    # ---Kill portion of the z-sample data if required---
    zd.kill_data(params.z_kill_list)
    
    # ---Load the pre-generated conformal blocks for long multiplets---
    #blocks = utils.generate_block_list(max(params.spin_list), params.z_kill_list)

    # ---Instantiate the crossing_eqn class---
    cft = BPS_SAC(params, zd)

    teor_reward = cft.best_theoretical_reward
    # array_index is the cluster array number passed to the console. Set it to zero if it doesn't exist.
    try:
        array_index = args.array_index
    except IndexError:
        array_index = 0

    # form the file_name where the code output is saved to
    file_name = params.filename_stem + str(array_index) + '.csv'
    utils.output_to_file(file_name=file_name, output=[teor_reward])
    # determine initial starting point in the form needed for the soft_actor_critic function
    x0 = params.global_best - params.shifts
    
    # ---Run the soft actor critic algorithm---
    soft_actor_critic(func=cft.crossing,
                      max_window_changes=params.max_window_exp,
                      window_decrease_rate=params.window_rate,
                      pc_max=params.pc_max,
                      file_name=file_name,
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
                      verbose=params.verbose)
