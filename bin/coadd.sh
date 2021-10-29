#! /bin/bash

brick=$1

outdir=/global/cscratch1/sd/dstn/dr10-early-coadds

export LEGACY_SURVEY_DIR=/global/cfs/cdirs/cosmo/work/legacysurvey/dr10

export GAIA_CAT_DIR=/global/cfs/cdirs/desi/target/gaia_edr3/healpix
export GAIA_CAT_PREFIX=healpix
export GAIA_CAT_SCHEME=nested
export GAIA_CAT_VER=E

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

bri=$(echo $brick | head -c 3)
mkdir -p $outdir/logs/$bri
log="$outdir/logs/$bri/$brick.log"

mkdir -p $outdir/metrics/$bri

echo Logging to: $log

python -u legacypipe/runbrick.py \
       --brick $brick \
       --bands g,r,i,z \
       --survey-dir $LEGACY_SURVEY_DIR \
       --outdir $outdir \
       --stage image_coadds \
       --blob-image --minimal-coadds \
       --no-write \
       --force-all \
       --skip-calibs \
       --nsatur 2 \
       --threads 8 \
       --old-calibs-ok \
       > $log 2>&1
