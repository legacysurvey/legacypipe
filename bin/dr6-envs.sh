#! /bin/bash

# setting up the environment variables for running the sweeps and external matching
# section of the pipeline for existing DR6 data
# see 
# https://github.com/legacysurvey/legacypipe/blob/master/doc/cookbook.md#sweeps-and-external-matching

# this file serves as a template; use in the sweep match jobs scripts.

# `source` this file (e.g. source dr6-envs.sh). Do not run it with bash.

set -x
export ATP_ENABLED=0

# for new DRs, these lines may need to be changed to the work directory rather than the
# data dir.
drdir=/global/projecta/projectdirs/cosmo/data/legacysurvey/dr6
outdir=$CSCRATCH/dr6-out

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

