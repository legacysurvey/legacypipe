#! /bin/bash

export LEGACY_SURVEY_DIR=/global/homes/d/dstn/legacypipe/py/euclid-dr1-dir

outdir=$LEGACY_SURVEY_DIR/zpt

export COSMO=/dvs_ro/cfs/cdirs/cosmo

export DUST_DIR=$COSMO/data/dust/v0_1

export GAIA_CAT_DIR=$COSMO/data/gaia/dr3/healpix
export GAIA_CAT_PREFIX=healpix
export GAIA_CAT_SCHEME=nested
export GAIA_CAT_VER=3

export TYCHO2_KD_DIR=$COSMO/staging/tycho2
export PS1CAT_DIR=$COSMO/work/ps1/cats/chunks-qz-star-v3
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

image_fn="$1"

export PYTHONPATH=.:${PYTHONPATH}

log=$outdir/logs/$(echo "${image_fn}" | sed s#.fits#.log#g)
logdir=$(dirname $log)
mkdir -p $logdir
echo "Logging to $log"

python -u -O legacyzpts/legacy_zeropoints.py \
       --camera vis \
       --threads 16 \
       --survey-dir ${LEGACY_SURVEY_DIR} \
       --outdir ${outdir} \
       --no-check-photom \
       $image_fn \
       >> "$log" 2>&1

