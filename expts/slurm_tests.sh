#!/bin/sh

#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name="tests"
#SBATCH --output=ccr/tests_pred.out
#SBATCH --error=ccr/tests_err.out
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
ulimit -s unlimited

echo "Launch Python job"
python -u scripts/test_convergence.py > ccr/test_convergence.out

#
echo "All Done!"
