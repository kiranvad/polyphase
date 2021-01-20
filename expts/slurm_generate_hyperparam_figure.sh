#!/bin/bash

#SBATCH --time=5:00:00
#SBATCH --cluster=faculty
#SBATCH --partition=planex --qos=planex
#SBATCH --account=olgawodo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --job-name="JOB"
#SBATCH --output=ccr/PRED.out
#SBATCH --error=ccr/ERR.out
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

echo "Launch Python job"
python -u scripts/generate_hyperparam_figure.py > ccr/OUT.out
echo "All Done!"
exit
