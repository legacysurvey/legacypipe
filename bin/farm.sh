#!/bin/bash

export LEGACY_SURVEY_DIR=$CSCRATCH/dr10a
outdir=$LEGACY_SURVEY_DIR/out-v3-cut

# Don't add ~/.local/ to Python's sys.path
export PYTHONNOUSERSITE=1
# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
# To avoid problems with MPI and Python multiprocessing
export MPICH_GNI_FORK_MODE=FULLCOPY
export KMP_AFFINITY=disabled

python -u $LEGACYPIPE_DIR/legacypipe/farm.py \
       --big drop \
       --pickle "${outdir}/pickles/%(brick).3s/runbrick-%(brick)s-srcs.pickle" \
       --inthreads 4 \
       --checkpoint "${outdir}/checkpoints/%(brickpre)s/checkpoint-%(brick)s.pickle" \
       farm
