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

ncores=4
# 128 GB / Cori Haswell node = 134217728 kbytes
maxmem=134217728
let usemem=${maxmem}*${ncores}/32
# Can detect Cori KNL node (96 GB) via:
# grep -q "Xeon Phi" /proc/cpuinfo && echo Yes
ulimit -Sv $usemem


echo "Logging to $logfile"
python -u legacypipe/forced_photom.py --survey-dir $DIR --catalog-dir-north $DIR/north --catalog-dir-south $DIR/south --catalog-resolve-dec-ngc 32.375 \
    --skip-calibs --apphot --derivs --camera $camera \
    $expnum $ccdname $outdir/$outfn > $logfile 2>&1

