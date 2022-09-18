import argparse
import os
import ray
from myexperiment import myexp # this is the main experiment function to call

parser = argparse.ArgumentParser()

parser.add_argument('--num_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--max_cpus', type=int, default=1, help='Maximum number of CPUs')
parser.add_argument('--cpus_per_job', type=int, default=1, help='Maximum number of CPUs per job')

args = parser.parse_args()

try:
    if os.environ.get('ip_head') is not None:
        assert os.environ.get('redis_password') is not None, "Missing redis password"
        ray.init(address=os.environ.get('ip_head'), _redis_password=os.environ.get('redis_password'))
        print("Connected to Ray cluster.")
        print(f"Available nodes: {ray.nodes()}")


    assert ray.is_initialized(), "Error in initializing ray."

    @ray.remote(num_cpus=args.cpus_per_job)
    def run_exp(argum):
        myexp(argum)

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

