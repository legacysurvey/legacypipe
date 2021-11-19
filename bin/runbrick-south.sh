#! /bin/bash

# Script for running the legacypipe code within a Shifter container at NERSC

# Burst-buffer!
#if [ "x$DW_PERSISTENT_STRIPED_DR9" == x ]; then
# No burst buffer -- use scratch

outdir=/global/cscratch1/sd/dstn/dr10-test

BLOB_MASK_DIR=/global/cfs/cdirs/cosmo/work/legacysurvey/dr8/south

export LEGACY_SURVEY_DIR=/global/cfs/cdirs/cosmo/work/legacysurvey/dr10
export CACHE_DIR=/global/cscratch1/sd/dstn/dr10-cache

#export GAIA_CAT_DIR=/global/cfs/cdirs/desi/target/gaia_edr3/healpix
export GAIA_CAT_DIR=/global/cscratch1/sd/adamyers/gaia_edr3/healpix
export GAIA_CAT_PREFIX=healpix
export GAIA_CAT_SCHEME=nested
export GAIA_CAT_VER=E

export DUST_DIR=/global/cfs/cdirs/cosmo/data/dust/v0_1
export UNWISE_COADDS_DIR=/global/cfs/cdirs/cosmo/work/wise/outputs/merge/neo7/fulldepth:/global/cfs/cdirs/cosmo/data/unwise/allwise/unwise-coadds/fulldepth
export UNWISE_COADDS_TIMERESOLVED_DIR=/global/cfs/cdirs/cosmo/work/wise/outputs/merge/neo7
export UNWISE_MODEL_SKY_DIR=/global/cfs/cdirs/cosmo/work/wise/unwise_catalog/dr3/mod

export TYCHO2_KD_DIR=/global/cfs/cdirs/cosmo/staging/tycho2
export LARGEGALAXIES_CAT=/global/cfs/cdirs/cosmo/staging/largegalaxies/v3.0/SGA-ellipse-v3.0.kd.fits
export SKY_TEMPLATE_DIR=/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/calib/sky_pattern

# Don't add ~/.local/ to Python's sys.path
export PYTHONNOUSERSITE=1
# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
# To avoid problems with MPI and Python multiprocessing
export MPICH_GNI_FORK_MODE=FULLCOPY
export KMP_AFFINITY=disabled

ncores=8

brick="$1"

bri=$(echo "$brick" | head -c 3)
mkdir -p "$outdir/logs/$bri"
log="$outdir/logs/$bri/$brick.log"

mkdir -p "$outdir/metrics/$bri"
mkdir -p "$outdir/pickles/$bri"

echo Logging to: "$log"
echo Running on $(hostname)

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

python -O $LEGACYPIPE_DIR/legacypipe/runbrick.py \
     --brick "$brick" \
     --skip \
     --skip-calibs \
     --bands g,r,i,z \
     --rgb-stretch 1.5 \
     --nsatur 2 \
     --survey-dir "$LEGACY_SURVEY_DIR" \
     --cache-dir "$CACHE_DIR" \
     --outdir "$outdir" \
     --blob-mask-dir "${BLOB_MASK_DIR}" \
     --checkpoint "${outdir}/checkpoints/${bri}/checkpoint-${brick}.pickle" \
     --pickle "${outdir}/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
     --threads "${ncores}" \
     >> "$log" 2>&1

#     --write-stage srcs \



#--run south \
#     --ps "${outdir}/metrics/${bri}/ps-${brick}-${SLURM_JOB_ID}.fits" \
#     --ps-t0 $(date "+%s") \

# Save the return value from the python command -- otherwise we
# exit 0 because the rm succeeds!
status=$?

# /Config directory nonsense
rm -R $TMPCACHE

exit $status


# QDO_BATCH_PROFILE=cori-shifter qdo launch -v tst 1 --cores_per_worker 8 --walltime=30:00 --batchqueue=debug --keep_env --batchopts "--image=docker:dstndstn/legacypipe:intel" --script "/src/legacypipe/bin/runbrick-shifter.sh"
