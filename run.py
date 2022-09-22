import argparse
import os
import ray
import itertools

parser = argparse.ArgumentParser()

parser.add_argument('--num_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--max_cpus', type=int, default=1, help='Maximum number of CPUs')
parser.add_argument('--cpus_per_job', type=int, default=1, help='Maximum number of CPUs per job')

args = parser.parse_args()

try:
    ray.init(address='auto', _node_ip_address="172.16.18.254")
    print("Connected to Ray cluster.")
    print(f"Available nodes: {ray.nodes()}")

    assert ray.is_initialized(), "Error in initializing ray."

    @ray.remote(num_cpus=args.cpus_per_job)
    def run_exp(argum):
        main(argum)

    parser.add_argument('--faff_max', type=int, default=200, help='Maximum number of steps without improving')
    parser.add_argument('--pc_max', type=int, default=20, help='Maximum number of reinitializations before reducing window')
    parser.add_argument('--window_rate', type=float, default=0.7, help='Rate of search window reduction')
    parser.add_argument('--max_window_exp', type=int, default=10, help='Maximun number of window reductions')
    parser.add_argument('--same_spin_hierarchy', type=bool, default=True, help='Whether same spin deltas should be ordered')
    parser.add_argument('--dyn_shift', type=float, default=0.3, help='Minimum distance between same spin deltas')
    parser.add_argument('--alpha', type=float, default=0.0005, help='Learning rate for actor network')
    parser.add_argument('--beta', type=float, default=0.0005, help='Learning rate for critic and value network')
    parser.add_argument('--reward_scale', type=float, default=7.0, help='The reward scale, also related to the entropy parameter')
    parser.add_argument('--gamma', type=float, default=0.99, help='Gamma parameter for cumulative reward')
    parser.add_argument('--tau', type=float, default=0.3, help='Tau parameter for state-value function update')
    parser.add_argument('--layer1_size', type=int, default=256, help='Dense units for first layer')
    parser.add_argument('--layer2_size', type=int, default=256, help='Dense units for the second layer')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

    faffs = [200, 400, 600, 800, 1000]
    pcs = [10, 20, 30, 40, 50]
    rates = [0.3, 0.5, 0.7, 0.9]
    windows = [5, 10, 15, 20, 25]
    shifts = [0.1, 0.3, 0.5, 0.7]
    scales = [1., 5., 10., 20.,]
    gammas = [0.8, 0.85, 0.9, 0.95, 0.99]
    taus = [0.0005, 0.005, 0.05]

    params = list(itertools.product(faffs, pcs, rates, windows, shifts, scales, gammas, taus))
    print(params)

    remaining_ids = []
    remaining_ids.append(run_exp.remote(args))
    n_jobs = len(remaining_ids)
    print(f"Total jobs: {n_jobs}")
    print(remaining_ids)

    # wait for jobs
    while remaining_ids:
        done_ids, remaining_ids = ray.wait(remaining_ids, num_returns=1)
        for result_id in done_ids:
            # There is only one return result by default.
            result = ray.get(result_id)
            n_jobs -= 1
            print(f'Job {result_id} terminated.\nJobs left: {n_jobs}')


finally:
    print('Shutting down ray...')
    ray.shutdown()
    print('Ray closed.')

