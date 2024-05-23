#! /bin/bash

#echo "runcalibs.sh on $(hostname)"

#outdir=$LEGACY_SURVEY_DIR/zpt
outdir=$SCRATCH/hsc-zpt

# Perlmutter
export COSMO=/dvs_ro/cfs/cdirs/cosmo
#export COSMO=/global/cfs/cdirs/cosmo

#export LEGACY_SURVEY_DIR=/global/cfs/cdirs/cosmo/work/users/dstn/ODIN/
export LEGACY_SURVEY_DIR=$COSMO/work/users/dstn/ODIN/

export GAIA_CAT_DIR=$COSMO/data/gaia/edr3/healpix
export GAIA_CAT_PREFIX=healpix
export GAIA_CAT_SCHEME=nested
export GAIA_CAT_VER=E

export DUST_DIR=$COSMO/data/dust/v0_1
export TYCHO2_KD_DIR=$COSMO/staging/tycho2
export LARGEGALAXIES_CAT=$COSMO/staging/largegalaxies/v3.0/SGA-ellipse-v3.0.kd.fits

export PS1CAT_DIR=$COSMO/work/ps1/cats/chunks-qz-star-v3

# DECam
#export SKY_TEMPLATE_DIR=/global/cfs/cdirs/cosmo/work/legacysurvey/dr9k/calib/sky_pattern

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

# NOTE, we do NOT set PYTHONPATH -- it is set in the Docker container.
# Specifically, DO NOT override it with a local checkout of legacypipe or any other package!
#echo "PYTHONPATH is $PYTHONPATH"

# # # Config directory nonsense
# export TMPCACHE=$(mktemp -d)
# mkdir $TMPCACHE/cache
# mkdir $TMPCACHE/config
# # astropy
# export XDG_CACHE_HOME=$TMPCACHE/cache
# export XDG_CONFIG_HOME=$TMPCACHE/config
# mkdir $XDG_CACHE_HOME/astropy
# cp -r $HOME/.astropy/cache $XDG_CACHE_HOME/astropy
# mkdir $XDG_CONFIG_HOME/astropy
# cp -r $HOME/.astropy/config $XDG_CONFIG_HOME/astropy
# # matplotlib
# export MPLCONFIGDIR=$TMPCACHE/matplotlib
# mkdir $MPLCONFIGDIR
# cp -r $HOME/.config/matplotlib $MPLCONFIGDIR
# # # ipython
# # export IPYTHONDIR=$TMPCACHE/ipython
# # mkdir $IPYTHONDIR
# # cp -r $HOME/.ipython $IPYTHONDIR

image_fns="$@"

first="$1"
#echo "First arg: $first"

camera=hsc

# Redirect logs to a nested directory.
#cpdir=$(basename $(dirname "${image_fn}"))
#logdir=$outdir/logs/$camera/$cpdir
#mkdir -p "$logdir"
#log=$(echo $(basename "${image_fn}" | sed s#.fits.fz#.log#g))
#log=$logdir/$log

log=$outdir/logs/$(echo "${first}" | sed s#.fits#.log#g)
logdir=$(dirname $log)
mkdir -p $logdir

echo "Logging to $log"

python -O $LEGACYPIPE_DIR/legacyzpts/legacy_zeropoints.py \
	--camera ${camera} \
    --survey-dir ${LEGACY_SURVEY_DIR} \
    --outdir ${outdir} \
    --no-check-photom \
    $image_fns \
    >> "$log" 2>&1

# # /Config directory nonsense
# rm -R $TMPCACHE
