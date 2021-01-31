#!/bin/sh

#SBATCH --time=10:00:00
#SBATCH --cluster=faculty
#SBATCH --partition=planex --qos=planex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --job-name="hamming"
#SBATCH --output=ccr/pred_hamming.out
#SBATCH --error=ccr/err_hamming.out
#SBATCH --mail-user=kiranvad@buffalo.edu
#SBATCH --mail-type=END
#SBATCH --exclusive
#SBATCH --requeue

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

module use /projects/academic/olgawodo/kiranvad/modulefiles
module load python/mypython37
source ~/.venv/polyphase/bin/activate
ulimit -s unlimited

echo "Launch Python job"
python -u scripts/compute_hamming.py > ccr/hamming.out

#
echo "All Done!"
