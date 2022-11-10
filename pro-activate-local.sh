#!/bin/bash
#SBATCH --job-name=ray-tune-trenta
### Modify this according to your Ray workload.
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --tasks-per-node=90
### Modify this according to your Ray workload.
#SBATCH --cpus-per-task=1
#SBATCH --output=ray.log
### Similarly, you can also specify the number of GPUs per node.
### Modify this according to your Ray workload. Sometimes this
### should be 'gres' instead.
#SBATCH --gpus-per-task=0

conda activate ray

# Getting the node names
nodes=$(sinfo -hN --state=idle|awk '{print $1}')
nodes_array=($nodes)

head_node="localhost"
head_node_ip_pro=172.16.18.254
export head_node_ip_pro

echo $head_node
# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
    echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port_pro=6380
ip_head_pro=$head_node_ip_pro:$port_pro
export ip_head_pro
echo "IP Head: $ip_head_pro"
export head_node_ip_pro
RAY_worker_register_timeout_seconds_pro=240
export RAY_worker_register_timeout_seconds_pro

echo "Starting HEAD at $head_node"
ray start --head --node-ip-address=$head_node_ip_pro --port=$port_pro --dashboard-port=8266

# optional, though may be useful in certain versions of Ray < 1.0.
# number of nodes other than the head node
worker_num=4

for ((i = 0; i < worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w $node_i \
        ray start --address $ip_head_pro \
        --block &
    sleep 5
done

# ray/doc/source/cluster/examples/simple-trainer.py
python -u pro_run_cluster.py