#! /bin/bash

brick=$1

outdir=/global/cscratch1/sd/dstn/dr10-early-coadds

export LEGACY_SURVEY_DIR=/global/cfs/cdirs/cosmo/work/legacysurvey/dr10

export CACHE_DIR=/global/cscratch1/sd/dstn/dr10-cache

#export GAIA_CAT_DIR=/global/cfs/cdirs/desi/target/gaia_edr3/healpix
export GAIA_CAT_DIR=/global/cscratch1/sd/adamyers/gaia_edr3/healpix
export GAIA_CAT_PREFIX=healpix
export GAIA_CAT_SCHEME=nested
export GAIA_CAT_VER=E

#export TYCHO2_KD_DIR=/global/cfs/cdirs/cosmo/staging/tycho2
#export LARGEGALAXIES_CAT=/global/cfs/cdirs/cosmo/staging/largegalaxies/v3.0/SGA-ellipse-v3.0.kd.fits
#export SKY_TEMPLATE_DIR=/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/calib/sky_pattern
export TYCHO2_KD_DIR=$CACHE_DIR/tycho2
export LARGEGALAXIES_CAT=$CACHE_DIR/SGA-ellipse-v3.0.kd.fits
export SKY_TEMPLATE_DIR=$CACHE_DIR/dr9-sky-pattern


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


bri=$(echo $brick | head -c 3)
mkdir -p $outdir/logs/$bri
log="$outdir/logs/$bri/$brick.log"

mkdir -p $outdir/metrics/$bri
#mkdir -p $outdir/pickles/$bri

echo Logging to: $log

python -O $LEGACYPIPE_DIR/legacypipe/runbrick.py \
       --brick $brick \
       --bands g,r,i,z \
       --rgb-stretch 1.5 \
       --survey-dir $LEGACY_SURVEY_DIR \
       --cache-dir $CACHE_DIR \
       --outdir $outdir \
       --skip-coadd \
       --stage image_coadds \
       --blob-mask \
       --minimal-coadds \
       --skip-calibs \
       --nsatur 2 \
       --force-all \
       --no-write \
       --cache-outliers \
       --threads 4 \
       > $log 2>&1

#       --write-stage halos \
#       --pickle "$outdir/pickles/$bri/runbrick-%(brick)s-%%(stage)s.pickle" \
#--no-write \
#       --force-all \
#       --stage tims --plots \
#       --zoom 1000 2500 2000 3000 \

#       --no-write \
    ##--threads 4 \
    #       --no-write \

#       --cache-dir /tmp/cache \
#       --min-mjd 56858.2232735 \
#       --max-mjd 56858.2232737 \

# Save the return value from the python command -- otherwise we
# exit 0 because the rm succeeds!
status=$?

#echo "max_memory $(cat /sys/fs/cgroup/memory/slurm/uid_$SLURM_JOB_UID/job_$SLURM_JOB_ID/memory.max_usage_in_bytes)" >> $log

# /Config directory nonsense
rm -R $TMPCACHE

exit $status
