#!/bin/bash

###Options
WORKER_SCRIPT='/global/cscratch1/sd/ziyaoz/farm/legacypipe.bac/py/legacypipe/worker.py'
###

###Input Validation
if [ -z "$1" ]
  then
    echo "Please specify farm address"
    exit 1
fi
###

###Dependencies
# Don't add ~/.local/ to Python's sys.path
export PYTHONNOUSERSITE=1
source /global/cscratch1/sd/ziyaoz/farm/mpi_bugfix.sh
###

python -O -u $WORKER_SCRIPT --threads 68 $1

###If you want another copy of the logs
#python -O -u $WORKER_SCRIPT --threads 68 $1 > worker-${SLURM_JOB_ID}-${SLURM_NODE_ID}-$(hostname).log
