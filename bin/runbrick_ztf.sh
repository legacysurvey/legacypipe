#!/bin/bash -l

# This script is run to process a single brick on a ZTF coadd


export CODEPATH=/project/projectdirs/uLens/code/bin
export PROJECTPATH=/global/homes/c/cwar4677 #/project/projectdirs/uLens/ZTF/Tractor/ #/global/homes/c/cwar4677

#cd $PROJECTPATH/legacypipe/py
#source $PROJECTPATH/legacypipe/bin/legacypipe-env
#cd $PROJECTPATH



source $CODEPATH/legacypipe/bin/legacypipe-env

export LEGACY_SURVEY_DIR=/project/projectdirs/uLens/ZTF/Tractor/data/ZTF18aakxvxm/G_small/tractor

#export PYTHONPATH=/project/projectdirs/uLens/ZTF/Tractor/legacypipe/py:$PYTHONPATH
export PYTHONPATH=$PROJECTPATH/legacypipe/py:$PYTHONPATH
export outdir=/global/homes/c/cwar4677/output_individual
#python $PROJECTPATH/legacypipe/py/ztfcoadd/ztfcoaddmaker.py --folder=$LEGACY_SURVEY_DIR  
python $PROJECTPATH/legacypipe/py/ztfcoadd/ztfCCDtablemaker.py $LEGACY_SURVEY_DIR $outdir
python $PROJECTPATH/legacypipe/py/legacypipe/runbrick.py --outdir=$outdir --pixscale=1.01 --no-wise --force-all --brick=2395p525 --threads=16 --plots
# --threads=32
#--radec=239.858822,52.209818
#--nblobs=50 --blob=750 --brick=2395p525 
