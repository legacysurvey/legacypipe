#! /bin/bash

# This script is run to process a single brick on a ZTF coadd

source /project/projectdirs/uLens/ZTF/Tractor/legacypipe/bin/legacypipe-env

export LEGACY_SURVEY_DIR=/project/projectdirs/uLens/ZTF/Tractor/data/ZTF18aakxvxm/G_coadd
outdir=$LEGACY_SURVEY_DIR
cd $outdir
export PYTHONPATH=/project/projectdirs/uLens/ZTF/Tractor/legacypipe/py:$PYTHONPATH

python /project/projectdirs/uLens/ZTF/Tractor/legacypipe/py/ztfcoadd/ztfCCDtablemaker.py $outdir
python /project/projectdirs/uLens/ZTF/Tractor/legacypipe/py/legacypipe/runbrick.py --brick=2395p525 --outdir=$outdir --pixscale=1.01 --no-wise --threads=32

