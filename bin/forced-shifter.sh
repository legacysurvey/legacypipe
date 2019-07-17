#! /bin/bash

DIR=/global/project/projectdirs/cosmo/work/legacysurvey/dr8

# Don't add ~/.local/ to Python's sys.path
export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY

camera=$1
expnum=$2
ccdname=$3
outfn=$4

exppre=$(echo $expnum | cut -c 1-3)

outdir=$CSCRATCH/dr8-forced

logdir=$outdir/forced/logs/$camera-$exppre
mkdir -p $logdir
logfile=$logdir/$camera-$expnum-$ccdname.log

cd /src/legacypipe/py

echo "Logging to $logfile"
python legacypipe/forced_photom.py --survey-dir $DIR --catalog-dir-north $DIR/north --catalog-dir-south $DIR/south --catalog-resolve-dec 32.375 \
    --skip-calibs --apphot --derivs --camera $camera \
    --threads 8 \
    $expnum $ccdname $outdir/$outfn > $logfile 2>&1

