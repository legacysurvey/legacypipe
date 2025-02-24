#! /bin/bash

img=$1

survey_dir=cfis-dir
out_dir=$SCRATCH/cfis-cosmos-u

export COSMO=/dvs_ro/cfs/cdirs/cosmo

export GAIA_CAT_DIR=$COSMO/data/gaia/edr3/healpix
export GAIA_CAT_PREFIX=healpix
export GAIA_CAT_SCHEME=nested
export GAIA_CAT_VER=E

export DUST_DIR=$COSMO/data/dust/v0_1
export TYCHO2_KD_DIR=$COSMO/staging/tycho2
export LARGEGALAXIES_CAT=$COSMO/staging/largegalaxies/v3.0/SGA-ellipse-v3.0.kd.fits

export PS1CAT_DIR=$COSMO/work/ps1/cats/chunks-qz-star-v3

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

log=$out_dir/zpt-logs/$(echo $img | sed s/.fz//g | sed s/.fits/.log/g)
logdir=$(dirname $log)
mkdir -p $logdir
echo "Logging to $log"

export PYTHONPATH=.:${PYTHONPATH}

python -u legacyzpts/legacy_zeropoints.py \
       --camera megaprime \
       --survey-dir ${survey_dir} \
       --outdir ${out_dir} \
       --image $img \
       --sdss-photom \
       --threads 10 \
       >> $log 2>&1
#       --force-cfht-ccds \

#       --plots --verboseplots \
#       --choose_ccd 1 \
