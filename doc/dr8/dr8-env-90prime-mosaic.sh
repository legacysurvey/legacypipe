#! /bin/bash
# Set up the software for running DR8.

export LEGACY_SURVEY_DIR=/global/project/projectdirs/cosmo/work/legacysurvey/dr8b/runbrick-90prime-mosaic

# Use local check-outs of legacypipe and legacyzpts.
export LEGACYPIPE_DIR=$LEGACY_SURVEY_DIR/code/legacypipe
export LEGACYZPTS_DIR=$LEGACY_SURVEY_DIR/code/legacyzpts

# This file has most of the needed environment variables.
source $LEGACYPIPE_DIR/bin/legacypipe-env

export PATH=$LEGACYPIPE_DIR/bin:$PATH
export PATH=$LEGACYZPTS_DIR/bin:$PATH
export PYTHONPATH=$LEGACYPIPE_DIR/py:$PYTHONPATH
export PYTHONPATH=$LEGACYZPTS_DIR/py:$PYTHONPATH

# Temporary, until we get the new conda stack working
export QDO_DIR=/global/project/projectdirs/cosmo/work/legacysurvey/dr8b/code/qdo
export PATH=$QDO_DIR/bin:$PATH
export PYTHONPATH=$QDO_DIR:$PYTHONPATH
if [ "$NERSC_HOST" = "edison" ]; then
    export PATH=/global/project/projectdirs/cosmo/work/legacysurvey/dr8b/code/build/$NERSC_HOST/bin:$PATH
    export PYTHONPATH=/global/project/projectdirs/cosmo/work/legacysurvey/dr8b/code/build/$NERSC_HOST/lib/python3.6/site-packages:$PYTHONPATH
fi    

# Some NERSC-specific options to get MPI working properly.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY

