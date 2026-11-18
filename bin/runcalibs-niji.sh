#! /bin/bash

export LEGACY_SURVEY_DIR=/data/dstn/niji-dir

outdir=$SCRATCH/niji-zpt

export GAIA_CAT_DIR=/data/dstn/gaia-dr3
export GAIA_CAT_PREFIX=healpix
export GAIA_CAT_SCHEME=nested
export GAIA_CAT_VER=3

export DUST_DIR=/data/dstn/dust-v0_1
export TYCHO2_KD_DIR=/data/dstn/tycho2
export PS1CAT_DIR=/data/dstn/ps1/chunks-qz-star-v3

unset LARGEGALAXIES_CAT
unset SKY_TEMPLATE_DIR
unset BLOB_MASK_DIR

# Don't add ~/.local/ to Python's sys.path
export PYTHONNOUSERSITE=1
# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
# To avoid problems with MPI and Python multiprocessing
export MPICH_GNI_FORK_MODE=FULLCOPY
export KMP_AFFINITY=disabled

image_fns="$@"

first="$1"

camera=hsc

log=$outdir/logs/$(echo "${first}" | sed s#.fits#.log#g)
logdir=$(dirname $log)
mkdir -p $logdir
echo "Logging to $log"

export LEGACYPIPE_DIR=/home/dstn/legacypipe/py

python -u -O $LEGACYPIPE_DIR/legacyzpts/legacy_zeropoints.py \
       --camera ${camera} \
       --survey-dir ${LEGACY_SURVEY_DIR} \
       --outdir ${outdir} \
       --no-check-photom \
       $image_fns \
       >> "$log" 2>&1

