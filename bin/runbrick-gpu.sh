#! /bin/bash

# Script for running the legacypipe code within a Shifter container at NERSC

export COSMO=/dvs_ro/cfs/cdirs/cosmo

export LEGACY_SURVEY_DIR=$COSMO/work/legacysurvey/dr11
outdir=$SCRATCH/dr11-gpu
#outdir=$SCRATCH/dr11-nogpu

export GAIA_CAT_DIR=$COSMO/data/gaia/dr3/healpix
export GAIA_CAT_PREFIX=healpix
export GAIA_CAT_SCHEME=nested
export GAIA_CAT_VER=3

export DUST_DIR=$COSMO/data/dust/v0_1
export UNWISE_COADDS_DIR=$COSMO/data/unwise/neo7/unwise-coadds/fulldepth:$COSMO/data/unwise/allwise/unwise-coadds/fulldepth
export UNWISE_COADDS_TIMERESOLVED_DIR=$COSMO/work/wise/outputs/merge/neo7
export UNWISE_MODEL_SKY_DIR=$COSMO/data/unwise/neo7/unwise-catalog/mod

export TYCHO2_KD_DIR=$COSMO/staging/tycho2
export LARGEGALAXIES_CAT=$COSMO/staging/largegalaxies/v3.0/SGA-ellipse-v3.0.kd.fits
export SKY_TEMPLATE_DIR=$COSMO/work/legacysurvey/dr10/calib/sky_pattern

unset BLOB_MASK_DIR
unset PS1CAT_DIR
unset GALEX_DIR

# Don't add ~/.local/ to Python's sys.path
export PYTHONNOUSERSITE=1
# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
# To avoid problems with MPI and Python multiprocessing
export MPICH_GNI_FORK_MODE=FULLCOPY
export KMP_AFFINITY=disabled




brick=""
zoom=""
gpumode=""
gpu=""
threads=""
subblobs=""
blobid=""
md="cpu"
th=""

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -brick)
      brick="$2"
      shift # past argument
      shift # past value
      ;;
    -zoom)
      zoom="--zoom $2"
      shift # past argument
      shift # past value
      ;;
    -gpumode)
      gpumode="--gpumode $2"
      md="gpumode$2"
      shift # past argument
      shift # past value
      ;;
    -gpu)
      gpu="--use-gpu"
      shift # past argument
      ;;
    -sub-blobs)
      subblobs="--sub-blobs"
      shift # past argument
      ;;
    -blobid)
      blobid="--bid $2"
      shift # past argument
      shift # past value
      ;;
    -outdir)
      outdir="$2"
      shift # past argument
      shift # past value
      ;;
    -threads)
      threads="--threads $2"
      th="threads$2"
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      echo "Error: Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

#brick="$1"
# strip whitespace from front and back
#brick="${brick#"${brick%%[![:space:]]*}"}"
#brick="${brick%"${brick##*[![:space:]]}"}"
bri=${brick:0:3}

mkdir -p "$outdir/logs/$bri"
mkdir -p "$outdir/metrics/$bri"
mkdir -p "$outdir/pickles/$bri"
log="$outdir/logs/$bri/$brick-$md-$th.log"
echo Logging to: "$log"
#echo Running on $(hostname)

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

echo -e "\n\n\n" >> "$log"
echo "-----------------------------------------------------------------------------------------" >> "$log"
echo -e "\nStarting on $(hostname)\n" >> "$log"
echo "-----------------------------------------------------------------------------------------" >> "$log"

#echo "Running:"
#echo "python -u legacypipe/runbrick.py --brick "$brick" --zoom 0 200 0 200 --use-gpu --skip --skip-calibs --bands g,r,i,z --rgb-stretch 1.5 --nsatur 2 --survey-dir $LEGACY_SURVEY_DIR --outdir $outdir --checkpoint ${outdir}/checkpoints/${bri}/checkpoint-${brick}.pickle --checkpoint-period 120 --pickle \'${outdir}/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle\' --release 10099 --no-wise"
#python -c "from photutils.aperture import CircularAperture, aperture_photometry"

## Run with a local checkout of the tractor and legacypipe repositories...
#export TRACTOR_DIR=./tractor-git
#export LEGACYPIPE_DIR=./legacypipe-git/py
#export PYTHONPATH=$TRACTOR_DIR:$LEGACYPIPE_DIR:${PYTHONPATH}

#if [ $# -ge 2 ]; then
#  gpumode="$2"
#else
#  gpumode=0
#fi
echo "BRICK = $brick"
echo "GPUMODE = $gpumode"
echo "GPU = $gpu"
echo "ZOOM = $zoom"
echo "THREADS = $threads"
echo "SUBBLOBS = $subblobs"
echo "BLOBID = $blobid"
echo "LOG = $log"

#python -u $LEGACYPIPE_DIR/legacypipe/runbrick.py \
#python -u -m cProfile -o brick.pro $LEGACYPIPE_DIR/legacypipe/runbrick.py \
python -u $LEGACYPIPE_DIR/legacypipe/runbrick.py \
     --brick "$brick" \
        $zoom \
        $gpu \
        $gpumode \
        $threads \
	$subblobs \
	$blobid \
     --skip-calibs \
     --bands g,r,i,z \
     --rgb-stretch 1.5 \
     --nsatur 2 \
     --survey-dir "$LEGACY_SURVEY_DIR" \
     --outdir "$outdir" \
     --pickle "${outdir}/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
     --write-stage srcs \
     --release 10099 \
     --no-wise \
      >> "$log" 2>&1

#     --use-gpu \
#     --gpumode $gpumode \
#     --zoom 100 300 100 300 \
#     --checkpoint "${outdir}/checkpoints/${bri}/checkpoint-${brick}.pickle" \
#     --checkpoint-period 120 \

#     --plots \
#     --threads 32 \
#     --write-stage srcs \
#     --no-wise-ceres helps for very dense fields.

# Save the return value from the python command -- otherwise we
# exit 0 because the rm succeeds!
status=$?

# /Config directory nonsense
rm -R $TMPCACHE

exit $status
