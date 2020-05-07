#! /bin/bash

# Run legacy_zeropoints on a single image within a Shifter container at NERSC.

export LEGACY_SURVEY_DIR=/global/cfs/cdirs/cosmo/work/legacysurvey/dr9

# NOTE: if you only want to regenerate sky calibs, MUST create a symlink
# in $calibdir/psfex, eg to
#   /global/cfs/cdirs/cosmo/work/legacysurvey/dr9/calib/psfex

outdir=$CSCRATCH/dr9-north-calib
zptsdir=${outdir}/zpts
calibdir=${outdir}/calib
imagedir=${LEGACY_SURVEY_DIR}/images

blob_dir=/global/cfs/cdirs/cosmo/work/legacysurvey/dr8/north
# Zeropoint_dir is only required for star halo subtraction, which we're
# not using in the north.
#zeropoints_dir=${LEGACY_SURVEY_DIR}

# work/dr8/north includes:
#   coadd -> /global/cfs/cdirs/cosmo/data/legacysurvey/dr8/north/coadd/
#   metrics -> /global/cfs/cdirs/cosmo/data/legacysurvey/dr8/north/metrics/
#   survey-bricks.fits.gz -> /global/cfs/cdirs/cosmo/data/legacysurvey/dr8/survey-bricks.fits.gz

ncores=4

# Record the time this script started, for reporting python startup time.
starttime=$(date +%s.%N)

export DUST_DIR=/global/cfs/cdirs/cosmo/data/dust/v0_1
export GAIA_CAT_DIR=/global/cfs/cdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom-2
export GAIA_CAT_VER=2
export TYCHO2_KD_DIR=/global/cfs/cdirs/cosmo/staging/tycho2
export LARGEGALAXIES_CAT=/global/cfs/cdirs/cosmo/staging/largegalaxies/v7.0/LSLGA-v7.0.kd.fits
export PS1CAT_DIR=/global/cfs/cdirs/cosmo/work/ps1/cats/chunks-qz-star-v3

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

image_fn="$1"

# Get the camera from the filename
#echo $image_fn
if [[ $image_fn == *"90prime"* ]]; then
  camera=90prime
elif [[ $image_fn == *"mosaic"* ]]; then
  camera=mosaic
else
  echo 'Unable to get camera from file name!'
  exit 1
fi

# Redirect logs to a nested directory.
cpdir=`echo $(basename $(dirname ${image_fn}))`
logdir=$outdir/logs-calibs/$camera/$cpdir
mkdir -p $logdir
log=`echo $(basename ${image_fn} | sed s#.fits.fz#.log#g)`
log=$logdir/$log

cmd="python -u /src/legacypipe/py/legacyzpts/legacy_zeropoints.py \
	--camera ${camera} \
    --image ${image_fn} \
    --image_dir ${imagedir} \
    --outdir ${zptsdir} \
    --calibdir ${calibdir} \
    --threads ${ncores} \
    --overhead ${starttime} \
    --quiet \
    --run-calibs-only \
    --blob-mask-dir ${blob_dir}"

echo $cmd > $log
$cmd >> $log 2>&1
