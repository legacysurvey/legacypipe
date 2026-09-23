#! /bin/bash

export LEGACY_SURVEY_DIR=/global/cfs/cdirs/desi/users/dstn/cfht-quicklook-dir
outdir=$LEGACY_SURVEY_DIR/zpt

export COSMO=/dvs_ro/cfs/cdirs/cosmo

export DUST_DIR=$COSMO/data/dust/v0_1

export GAIA_CAT_DIR=$COSMO/data/gaia/dr3/healpix
export GAIA_CAT_PREFIX=healpix
export GAIA_CAT_SCHEME=nested
export GAIA_CAT_VER=3

export TYCHO2_KD_DIR=$COSMO/staging/tycho2
#export LARGEGALAXIES_CAT=$COSMO/staging/largegalaxies/v3.0/SGA-ellipse-v3.0.kd.fits
#export LARGEGALAXIES_CAT=$LEGACY_SURVEY_DIR/SGA-2020.kd.fits
unset LARGEGALAXIES_CAT
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

# # Config directory nonsense
export TMPCACHE=$(mktemp -d)
mkdir $TMPCACHE/cache
mkdir $TMPCACHE/config
# astropy
export XDG_CACHE_HOME=$TMPCACHE/cache
export XDG_CONFIG_HOME=$TMPCACHE/config
mkdir $XDG_CACHE_HOME/astropy
cp -r $HOME/.astropy/cache $XDG_CACHE_HOME/astropy
mkdir $XDG_CONFIG_HOME/astropy
cp -r $HOME/.astropy/config $XDG_CONFIG_HOME/astropy
# matplotlib
export MPLCONFIGDIR=$TMPCACHE/matplotlib
mkdir $MPLCONFIGDIR
cp -r $HOME/.config/matplotlib $MPLCONFIGDIR

image="$1"

camera=quicklook

log=$outdir/logs/$(echo "${image}" | sed s#.flt#.log#g)
logdir=$(dirname $log)
mkdir -p $logdir

echo "Logging to $log"

# Local
export LEGACYPIPE_DIR=/global/homes/d/dstn/legacypipe/py
export PYTHONPATH=$LEGACYPIPE_DIR:${PYTHONPATH}

python -O $LEGACYPIPE_DIR/legacyzpts/legacy_zeropoints.py \
	--camera ${camera} \
    --survey-dir ${LEGACY_SURVEY_DIR} \
    --outdir ${outdir} \
    $image \
    >> "$log" 2>&1

# Save the return value from the python command -- otherwise we
# exit 0 because the rm succeeds!
status=$?

# /Config directory nonsense
rm -R $TMPCACHE

exit $status
