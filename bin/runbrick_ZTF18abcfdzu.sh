#!/bin/bash -l

# This script is run to process a single brick on a ZTF coadd

#export CODEPATH=/project/projectdirs/uLens/code/bin
export PROJECTPATH=/global/homes/c/cwar4677/tractor_dr8
#source $PROJECTPATH/legacypipe/bin/legacypipe-env
#cd $PROJECTPATH/legacypipe/py
#source $PROJECTPATH/legacypipe/bin/legacypipe-env
#cd $PROJECTPATH

export LEGACY_SURVEY_DIR=/project/projectdirs/uLens/ZTF/Tractor/data/ZTF18abcfdzu/tractor

#export PYTHONPATH=/project/projectdirs/uLens/ZTF/Tractor/legacypipe/py:$PYTHONPATH
#export PYTHONPATH=/global/homes/c/cwar4677:$PYTHONPATH
export outdir=$LEGACY_SURVEY_DIR #/global/homes/c/cwar4677/output_ZTF18aaymybb
export PYTHONPATH=$PROJECTPATH/legacypipe/py:$PYTHONPATH
export PYTHONPATH=$PROJECTPATH/new_tractor/tractor-1:$PYTHONPATH

#python $PROJECTPATH/legacypipe/py/ztfcoadd/ztfcoaddmaker.py --folder=$LEGACY_SURVEY_DIR/images  
#python $PROJECTPATH/legacypipe/py/ztfcoadd/ztfCCDtablemaker.py $LEGACY_SURVEY_DIR $outdir

#python $PROJECTPATH/legacypipe/py/legacypipe/runbrick.py --outdir=$outdir --coadd-bw --nsigma=5 --stage fitblobs --radec 230.217170 54.215558 --blobradec 230.217170 54.215558 --unwise-dir $LEGACY_SURVEY_DIR/images --no-wise --old-calibs-ok #--plots

#python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 53820533 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_53820533.fits #$LEGACY_SURVEY_DIR/tractor/cus/tractor-custom-230217p54215.fits 

#--stage fitblobs
#--stage writecat
#-blobradec 288.656715 50.481882 
#--threads=32
#--nblobs=1 --blob=1112 
#--blob=340
#--blob=274
#--radec=239.858822,52.209818
#--nblobs=50 --blob=750 --brick=2395p525 

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 53820533 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_53820533.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 53820717 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_53820717.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 54019847 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_54019847.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 54019987 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_54019987.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 54022336 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_54022336.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 54022476 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_54022476.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 54422528 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_54422528.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 54520615 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_54520615.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 54522590 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_54522590.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 54630467 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_54630467.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 54728305 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_54728305.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 55020261 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_55020261.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 55321777 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_55321777.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 55324457 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_55324457.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 55332245 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_55332245.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 55419724 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_55419724.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 55421442 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_55421442.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 55724159 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_55724159.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 55833092 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_55833092.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 56617592 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_56617592.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 56619597 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_56619597.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 56718445 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_56718445.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 56928200 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_56928200.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 57120391 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_57120391.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 57217653 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_57217653.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 57219605 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_57219605.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 57319640 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_57319640.fits

python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits 57417945 CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_57417945.fits

