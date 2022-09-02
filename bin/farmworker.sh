#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Need one argument: address of server socket, eg tcp://cori06:5555"
    exit -1
fi

server=$1

# Don't add ~/.local/ to Python's sys.path
export PYTHONNOUSERSITE=1
# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
# To avoid problems with MPI and Python multiprocessing
export MPICH_GNI_FORK_MODE=FULLCOPY
export KMP_AFFINITY=disabled

python -u $LEGACYPIPE_DIR/legacypipe/worker.py \
       --threads 32 \
       $server
