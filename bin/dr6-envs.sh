#! /bin/bash

# setting up the environment variables for running the sweeps and external matching
# section of the pipeline for existing DR6 data
# see 
# https://github.com/legacysurvey/legacypipe/blob/master/doc/cookbook.md#sweeps-and-external-matching

set -x
export ATP_ENABLED=0

outdir=$CSCRATCH/dr6-out
# for new DRs, this line may need to be changed to the work directory rather than the
# data dir.
drdir=/global/projecta/projectdirs/cosmo/data/legacysurvey/dr6

export LEGACYPIPE_DIR=/global/cscratch1/sd/desiproc/DRcode/legacypipe

export TRACTOR_INDIR=$drdir/tractor
export BRICKSFILE=$drdir/survey-bricks.fits.gz
export TRACTOR_FILELIST=$outdir/tractor_filelist
export SWEEP_OUTDIR=$outdir/sweep
export PYTHONPATH=$LEGACYPIPE_DIR/py:${PYTHONPATH}

export EXTERNAL_OUTDIR=$CSCRATCH/$dr/external
export SDSSDIR=/global/projecta/projectdirs/sdss/data/sdss

set +x

/usr/bin/mkdir -p $outdir
/usr/bin/mkdir -p $EXTERNAL_OUTDIR
/usr/bin/mkdir -p $SWEEP_OUTDIR

if ! [ -f $TRACTOR_FILELIST ]; then
    find $TRACTOR_INDIR -name 'tractor-*.fits' > $TRACTOR_FILELIST
else
    echo $TRACTOR_FILELIST already exists.
    echo run the following command to rebuild the list of tractor tiles
    echo "find $TRACTOR_INDIR -name 'tractor-*.fits' > $TRACTOR_FILELIST"
fi

