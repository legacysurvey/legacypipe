#! /bin/bash

# This script is run to process a single brick on a ZTF coadd

# Assume we are running in the legacypipe/py directory.
cd /project/projectdirs/uLens/ZTF/Tractor/legacypipe/py

export LEGACY_SURVEY_DIR=/project/projectdirs/uLens/ZTF/Tractor/data/ZTF18aakxvxm/G_trim
outdir=$LEGACY_SURVEY_DIR

export PYTHONPATH=/project/projectdirs/uLens/ZTF/Tractor/legacypipe/py:$PYTHONPATH

source /project/projectdirs/uLens/ZTF/Tractor/legacypipe/bin/legacypipe-env

python /project/projectdirs/uLens/ZTF/Tractor/legacypipe/py/ztfcoadd/ztfCCDtablemaker.py $outdir
python /project/projectdirs/uLens/ZTF/Tractor/legacypipe/py/legacypipe/runbrick.py --brick=2395p525 --outdir=outdir --pixscale=1.01 --blobid=2142
