#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --cluster=faculty
#SBATCH --partition=planex --qos=planex
#SBATCH --account=olgawodo
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=20
#SBATCH --job-name="HTE"
#SBATCH --output=ccr/pred_hte.out
#SBATCH --error=ccr/err_hte.out
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

#for ray 
worker_num=4 # Must be one less that the total number of nodes

module use /projects/academic/olgawodo/kiranvad/modulefiles
module load python/mypython37
ulimit -s unlimited

chmod +x ray_start_head.sh
chmod +x ray_start_worker.sh

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node1=${nodes_array[0]}

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # making redis-address
suffix=':6379'
ip_head=$ip_prefix$suffix
echo "IP Head: $ip_head"

export ip_head

echo "STARTING HEAD at $node1"
srun --nodes=1 --ntasks=1 -w $node1 ray_start_head.sh &
sleep 15

for ((  i=1; i<=$worker_num; i++ ))
do
 node_i=${nodes_array[$i]}
 echo "STARTING WORKER $i at $node_i"
 srun --nodes=1 --ntasks=1 -w $node_i ray_start_worker.sh $ip_head $i &
 sleep 5
done

echo "Launch Python job"
python -u scripts/hte.py 80 > ccr/hte.out

echo "ENDING SLEEP"
pkill -P $(<./pid_storage/head.pid) sleep
for ((  i=1; i<=$worker_num; i++ ))
do
 pkill -P $(<./pid_storage/worker${i}.pid) sleep
done

echo "All Done!"
