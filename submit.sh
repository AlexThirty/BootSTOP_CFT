#!/bin/bash

#SBATCH --job-name=cl.cossu
#SBATCH --nodes=5
#SBATCH --tasks-per-node=1

worker_num=4  # this must always be equal to --nodes - 1


nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node1=${nodes_array[0]}

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)

export ip_head # Exporting for latter access by trainer.py
export redis_password
echo Head node ${node1}
echo IP of head node ${ip_head}

srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --port=6379 --redis-password=$redis_password & # Starting the head
sleep 10
# Make sure the head successfully starts before any worker does, otherwise
# the worker will not be able to connect to redis. In case of longer delay,
# adjust the sleeptime above to ensure proper order.

for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  echo Worker node ${node2}
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password & # Starting the workers
  # Flag --block will keep ray process alive on each compute node.
  sleep 10
done

echo "Launching python -u launch_exp.py $@"
python -u launch_exp.py $@
