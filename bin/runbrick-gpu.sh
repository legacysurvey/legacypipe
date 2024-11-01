#! /bin/bash

# Script for running the legacypipe code within a Shifter container at NERSC

export COSMO=/dvs_ro/cfs/cdirs/cosmo

export LEGACY_SURVEY_DIR=$COSMO/work/legacysurvey/dr11
outdir=$SCRATCH/dr11-gpu

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

brick="$1"
# strip whitespace from front and back
#brick="${brick#"${brick%%[![:space:]]*}"}"
#brick="${brick%"${brick##*[![:space:]]}"}"
bri=${brick:0:3}

mkdir -p "$outdir/logs/$bri"
mkdir -p "$outdir/metrics/$bri"
mkdir -p "$outdir/pickles/$bri"
log="$outdir/logs/$bri/$brick.log"
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

python -u legacypipe/runbrick.py \
     --brick "$brick" \
     --use-gpu \
     --skip \
     --skip-calibs \
     --bands g,r,i,z \
     --rgb-stretch 1.5 \
     --nsatur 2 \
     --survey-dir "$LEGACY_SURVEY_DIR" \
     --outdir "$outdir" \
     --checkpoint "${outdir}/checkpoints/${bri}/checkpoint-${brick}.pickle" \
     --checkpoint-period 120 \
     --pickle "${outdir}/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
     --release 10099 \
     --no-wise \
      >> "$log" 2>&1

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
