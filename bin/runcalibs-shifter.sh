#! /bin/bash

# Run legacy_zeropoints on a single image within a Shifter container at NERSC.

export LEGACY_SURVEY_DIR=/global/project/projectdirs/cosmo/work/legacysurvey/dr8

ncores=8

# Record the time this script started, for reporting python startup time.
starttime=$(date +%s.%N)

export DUST_DIR=/global/project/projectdirs/cosmo/data/dust/v0_1
export GAIA_CAT_DIR=/global/project/projectdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom-2
export GAIA_CAT_VER=2
export TYCHO2_KD_DIR=/global/project/projectdirs/cosmo/staging/tycho2
export LARGEGALAXIES_DIR=/global/project/projectdirs/cosmo/staging/largegalaxies/v2.0
export PS1CAT_DIR=/global/project/projectdirs/cosmo/work/ps1/cats/chunks-qz-star-v3

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
echo "PYTHONPATH is $PYTHONPATH"

outdir=${LEGACY_SURVEY_DIR}
zptsdir=${outdir}/zpts
calibdir=$outdir/calib
imagedir=${LEGACY_SURVEY_DIR}/images

image_fn="$1"

# Get the camera from the filename
echo $image_fn
if [[ $image_fn == *"decam"* ]]; then
  camera=decam
elif [[ $image_fn == *"90prime"* ]]; then
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
echo Logging to: $log

python /src/legacypipe/py/legacyzpts/legacy_zeropoints.py \
	--camera ${camera} \
    --image ${image_fn} \
    --image_dir ${imagedir} \
    --outdir ${zptsdir} \
    --calibdir ${calibdir} \
    --threads ${ncores} \
    --overhead ${starttime} \
    --run-calibs-only \
    >> $log 2>&1

