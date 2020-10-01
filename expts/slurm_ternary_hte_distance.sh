#!/bin/bash

#SBATCH --time=5:00:00
#SBATCH --cluster=ub-hpc
#SBATCH --partition=general-compute --qos=general-compute
#SBATCH --account=olgawodo
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=32
#SBATCH --job-name="dist"
#SBATCH --output=ccr/pred_dist.out
#SBATCH --error=ccr/err_dist.out
#SBATCH --mail-user=kiranvad@buffalo.edu
#SBATCH --mail-type=END
#SBATCH --exclusive
#SBATCH --requeue

# usual sbatch commands
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR

module use /projects/academic/olgawodo/kiranvad/modulefiles
module load python/mypython37
ulimit -s unlimited

################# DON NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###############
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
# I took it from https://github.com/NERSC/slurm-ray-cluster

redis_password=$(uuidgen)
export redis_password

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node_1=${nodes_array[0]} 
ip=$(srun --nodes=1 --ntasks=1 -w $node_1 hostname --ip-address) # making redis-address
port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w $node_1 ray_start_head.sh $ip $redis_password &
sleep 30

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((  i=1; i<=$worker_num; i++ ))
do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w $node_i ray_start_worker.sh $ip_head $redis_password &
  sleep 30
done
##############################################################################################

echo "Launch Python job"
python -u scripts/ternary_hte_distance.py > ccr/dist.out
echo "All Done!"
exit
