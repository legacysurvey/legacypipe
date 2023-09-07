#! /bin/bash

# Run legacy_zeropoints on a single image within a Shifter container at NERSC.

export COSMO=/global/cfs/cdirs/cosmo
export LEGACY_SURVEY_DIR=$COSMO/work/legacysurvey/dr11
#export CACHE_DIR=/global/cscratch1/sd/dstn/dr10-cache
#outdir=$LEGACY_SURVEY_DIR/zpt
outdir=$SCRATCH/zpt

export COSMO_RO=/dvs_ro/cfs/cdirs/cosmo


# NOTE: if you only want to regenerate sky calibs, MUST create a symlink
# in $LSD/calib/psfex, eg to
#   /global/cfs/cdirs/cosmo/work/legacysurvey/dr10/calib/psfex

# blob_dir=/global/cfs/cdirs/cosmo/work/legacysurvey/dr8-blobmask-south
# zeropoints_dir=${LEGACY_SURVEY_DIR}

# dr8-blobmask-south contains:
#   coadd -> /global/cfs/cdirs/cosmo/data/legacysurvey/dr8/south/coadd/
#   metrics -> /global/cfs/cdirs/cosmo/data/legacysurvey/dr8/south/metrics/
#   survey-bricks.fits.gz -> /global/cfs/cdirs/cosmo/data/legacysurvey/dr8/survey-bricks.fits.gz

#CACHE_DIR=/tmp/dr10pre-cache
#mkdir -p "${CACHE_DIR}"

#ncores=32
ncores=8

#export DUST_DIR=/global/cfs/cdirs/cosmo/data/dust/v0_1
#export TYCHO2_KD_DIR=/global/cfs/cdirs/cosmo/staging/tycho2
#export LARGEGALAXIES_CAT=/global/cfs/cdirs/cosmo/staging/largegalaxies/v3.0/SGA-ellipse-v3.0.kd.fits
export DUST_DIR=$COSMO_RO/data/dust/v0_1
export GAIA_CAT_DIR=$COSMO_RO/work/gaia/chunks-gaia-dr2-astrom-2
export GAIA_CAT_VER=2
unset GAIA_CAT_PREFIX
unset GAIA_CAT_SCHEME

export TYCHO2_KD_DIR=$COSMO_RO/staging/tycho2
export LARGEGALAXIES_CAT=$COSMO_RO/staging/largegalaxies/v3.0/SGA-ellipse-v3.0.kd.fits
export PS1CAT_DIR=$COSMO_RO/work/ps1/cats/chunks-qz-star-v3

#export SKY_TEMPLATE_DIR=/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/calib/sky_pattern
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

#log=$outdir/logs/$(echo "${image_fn}" | sed s#.fits.fz#.log#g)
log=$outdir/logs2/$(echo "${image_fn}" | sed s#.fits.fz#.log#g)

logdir=$(dirname $log)
mkdir -p $logdir
echo "Logging to $log"

#-o strace-$(echo ${image_fn} | sed s+/+-+g).log \
#strace -f -e trace=open,openat \
#
#python -u -O $LEGACYPIPE_DIR/legacyzpts/legacy_zeropoints.py \


#    --cache-dir ${CACHE_DIR} \
#    --prime-cache \

python -O $LEGACYPIPE_DIR/legacyzpts/legacy_zeropoints.py \
	--camera ${camera} \
    --survey-dir ${LEGACY_SURVEY_DIR} \
    --image ${image_fn} \
    --outdir ${outdir} \
    --threads ${ncores} \
    >> "$log" 2>&1

#    --fitsverify \
#    --verbose \
#    --blob-mask-dir ${blob_dir} \
#    --zeropoints-dir ${zeropoints_dir}"

# Save the return value from the python command -- otherwise we
# exit 0 because the rm succeeds!
status=$?

# /Config directory nonsense
rm -R $TMPCACHE

exit $status
