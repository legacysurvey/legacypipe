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

# This fails if we're not running in the container, but then the rest of the script works fine.
cd /src/legacypipe/py

echo "Logging to $logfile"
python legacypipe/forced_photom.py \
       --survey-dir $DIR \
       --catalog-dir-north $DIR/north --catalog-dir-south $DIR/south \
       --catalog-resolve-dec-ngc 32.375 \
       --skip-calibs --apphot --derivs --camera $camera \
       --threads 32 \
       $expnum $ccdname $outdir/$outfn > $logfile 2>&1

# eg:
# QDO_BATCH_PROFILE=cori-shifter qdo launch forced-south 1 --cores_per_worker 32 --walltime=30:00 --batchqueue=debug --batchopts "--image=docker:legacysurvey/legacypipe:nersc-dr9.0.1 --license=SCRATCH,project" --script "../bin/forced-south.sh" --keep_env

# Shared queue (--threads 16)
# qdo launch forced-south 1 --cores_per_worker 16 --walltime=1:00:00 --batchqueue=shared --batchopts "--license=SCRATCH,project -a 0-99 --cpus-per-task=32" --script "../bin/forced-south-16.sh" --keep_env -v
