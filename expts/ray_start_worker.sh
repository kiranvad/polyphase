#!/bin/bash
cd $SLURM_SUBMIT_DIR
ray start --address=$1

echo "Worker ${2} PID: $$"
echo "$$" | tee ./pid_storage/worker${2}.pid
sleep infinity

echo "Worker ${2} stopped"