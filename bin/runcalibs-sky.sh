#! /bin/bash

export COSMO=/dvs_ro/cfs/cdirs/cosmo

export LEGACY_SURVEY_DIR=$COSMO/work/legacysurvey/dr11
outdir=$SCRATCH/dr11

ncores=32

export GAIA_CAT_DIR=$COSMO/data/gaia/dr3/healpix
export GAIA_CAT_PREFIX=healpix
export GAIA_CAT_SCHEME=nested
export GAIA_CAT_VER=3

export DUST_DIR=$COSMO/data/dust/v0_1

export TYCHO2_KD_DIR=$COSMO/staging/tycho2
export LARGEGALAXIES_CAT=$COSMO/staging/largegalaxies/v3.0/SGA-ellipse-v3.0.kd.fits
export SKY_TEMPLATE_DIR=$COSMO/work/legacysurvey/dr11/calib/sky_pattern
export BLOB_MASK_DIR=$COSMO/work/legacysurvey/dr11-early-v2

unset PS1CAT_DIR

# Don't add ~/.local/ to Python's sys.path
export PYTHONNOUSERSITE=1
# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
# To avoid problems with MPI and Python multiprocessing
export MPICH_GNI_FORK_MODE=FULLCOPY
export KMP_AFFINITY=disabled

# NOTE, we do NOT set PYTHONPATH -- it is set in the Docker container.
# Specifically, DO NOT override it with a local checkout of legacypipe or any other package!
#echo "PYTHONPATH is $PYTHONPATH"

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
# # ipython
# export IPYTHONDIR=$TMPCACHE/ipython
# mkdir $IPYTHONDIR
# cp -r $HOME/.ipython $IPYTHONDIR

image_fn="$1"

camera=decam

# Redirect logs to a nested directory.
cpdir=$(basename $(dirname "${image_fn}"))
logdir=$outdir/logs/$camera/$cpdir
mkdir -p "$logdir"
log=$(echo $(basename "${image_fn}" | sed s#.fits.fz#.log#g))
log=$logdir/$log

echo "Logging to $log"

python -O $LEGACYPIPE_DIR/legacyzpts/legacy_zeropoints.py \
    --threads ${ncores} \
	--camera ${camera} \
    --survey-dir ${LEGACY_SURVEY_DIR} \
    --cache-dir ${CACHE_DIR} \
    --image ${image_fn} \
    --outdir ${outdir} \
    --run-sky-only \
    --run-calibs-only \
    --blob-mask-dir $BLOB_MASK_DIR \
    --zeropoints-dir $LEGACY_SURVEY_DIR \
    --calibdir ${outdir}/calib \
    --no-check-photom \
    >> "$log" 2>&1
status=$?

# /Config directory nonsense
rm -R $TMPCACHE

exit $status
