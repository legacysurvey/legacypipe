#! /bin/bash
# Set up the software for running DR8.  Note the assumption that locally
# checked-out code is in the "code" subdirectory relative to where this script
# is sourced!

CODE_DIR=$PWD/code

# Use local check-outs of legacypipe and legacyzpts.
export LEGACYPIPE_DIR=$CODE_DIR/legacypipe
export LEGACYZPTS_DIR=$CODE_DIR/legacyzpts

# This file has most of the needed environment variables.
source $LEGACYPIPE_DIR/bin/legacypipe-env

export PATH=$LEGACYPIPE_DIR/bin:$PATH
export PATH=$LEGACYZPTS_DIR/bin:$PATH
export PYTHONPATH=$LEGACYPIPE_DIR/py:$PYTHONPATH
export PYTHONPATH=$LEGACYZPTS_DIR/py:$PYTHONPATH

# Temporary, until we get the new conda stack working on edison.
export QDO_DIR=$CODE_DIR/qdo
export PATH=$QDO_DIR/bin:$PATH
export PYTHONPATH=$QDO_DIR:$PYTHONPATH
if [ "$NERSC_HOST" = "edison" ]; then
  export PATH=$CODE_DIR/build/bin:$PATH
  export PYTHONPATH=$CODE_DIR/build/lib/python3.6/site-packages:$PYTHONPATH
fi

# Some NERSC-specific options to get MPI working properly.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY

