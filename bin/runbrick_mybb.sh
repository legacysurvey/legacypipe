#!/bin/bash -l

# This script is run to process a single brick on a ZTF coadd


#export CODEPATH=/project/projectdirs/uLens/code/bin
export PROJECTPATH=/global/homes/c/cwar4677

#cd $PROJECTPATH/legacypipe/py
#source $PROJECTPATH/legacypipe/bin/legacypipe-env
#cd $PROJECTPATH



source $PROJECTPATH/legacypipe/bin/legacypipe-env

export LEGACY_SURVEY_DIR=/project/projectdirs/uLens/ZTF/Tractor/data/ZTF18aaymybb/tractor/

#export PYTHONPATH=/project/projectdirs/uLens/ZTF/Tractor/legacypipe/py:$PYTHONPATH
#export PYTHONPATH=/global/homes/c/cwar4677:$PYTHONPATH
export outdir=$LEGACY_SURVEY_DIR #/global/homes/c/cwar4677/output_ZTF18aaymybb
export PYTHONPATH=$PROJECTPATH/legacypipe/py:$PYTHONPATH
export PYTHONPATH=$PROJECTPATH:$PYTHONPATH

#python $PROJECTPATH/legacypipe/py/ztfcoadd/ztfcoaddmaker.py --folder=$LEGACY_SURVEY_DIR  
#python $PROJECTPATH/legacypipe/py/ztfcoadd/ztfCCDtablemaker.py $LEGACY_SURVEY_DIR $outdir

python $PROJECTPATH/legacypipe/py/legacypipe/runbrick.py --outdir=$outdir --no-wise --coadd-bw --nsigma=6 --stage fitblobs --radec 288.656715 50.481882 --blobradec 288.656715 50.481882 --no-gaia #--plots
#--stage fitblobs
#--stage writecat
#-blobradec 288.656715 50.481882 
#--threads=32
#--nblobs=1 --blob=1112 
#--blob=340
#--blob=274
#--radec=239.858822,52.209818
#--nblobs=50 --blob=750 --brick=2395p525 
