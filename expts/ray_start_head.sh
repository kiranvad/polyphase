#!/bin/bash
cd $SLURM_SUBMIT_DIR
ray start --head --redis-port=6379

echo "Head PID: $$"
echo "$$" | tee ./pid_storage/head.pid
sleep infinity

echo "Head stopped"