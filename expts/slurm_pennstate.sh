#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --cluster=faculty
#SBATCH --partition=planex --qos=planex
#SBATCH --account=olgawodo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --job-name="pennstate"
#SBATCH --output=ccr/pred_pennstate.out
#SBATCH --error=ccr/err_pennstate.out
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

#for ray 
worker_num=0 # Must be one less that the total number of nodes


# module load Langs/Python/3.6.4 # This will vary depending on your environment
# source venv/bin/activate
module use /projects/academic/olgawodo/kiranvad/modulefiles
module load python/mypython37
ulimit -s unlimited


nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node1=${nodes_array[0]}

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)

export ip_head # Exporting for latter access by trainer.py

srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --redis-port=6379 --redis-password=$redis_password & # Starting the head
sleep 15
# Make sure the head successfully starts before any worker does, otherwise
# the worker will not be able to connect to redis. In case of longer delay,
# adjust the sleeptime above to ensure proper order.

for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password & # Starting the workers
  # Flag --block will keep ray process alive on each compute node.
  sleep 5
done


echo "Launch Python job"
python -u scripts/pennstate_version2.py $redis_password 80 > ccr/pennstate.out

echo "All Done!"

