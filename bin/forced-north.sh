#! /bin/bash

DIR=/global/cfs/cdirs/cosmo/work/legacysurvey/dr9
outdir=$CSCRATCH/dr9-forced

# SGA galaxy and Gaia star halo subtraction
# (star halos are only for the south though)
export LARGEGALAXIES_CAT=/global/cfs/cdirs/cosmo/staging/largegalaxies/v3.2/SGA-ellipse-v3.2.kd.fits
export GAIA_CAT_DIR=/global/cfs/cdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom-2
export GAIA_CAT_VER=2
unset SKY_TEMPLATE_DIR

# Don't add ~/.local/ to Python's sys.path
export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY

camera=$1
expnum=$2

exppre=$(printf %08d "$expnum" | cut -c 1-5)

logdir=$outdir/logs/$camera/$exppre
mkdir -p "$logdir"
logfile=$logdir/$expnum.log

ncores=4
# 128 GB / Cori Haswell node = 134217728 kbytes
#maxmem=134217728
#let usemem=${maxmem}*${ncores}/32
# Can detect Cori KNL node (96 GB) via:
# grep -q "Xeon Phi" /proc/cpuinfo && echo Yes
#ulimit -Sv $usemem

echo "Logging to $logfile"

python -O $LEGACYPIPE_DIR/legacypipe/forced_photom.py \
       --survey-dir $DIR \
       --catalog-dir-north $DIR/north \
       --catalog-dir-south $DIR/south \
       --catalog-resolve-dec-ngc 32.375 \
       --skip \
       --skip-calibs \
       --apphot \
       --derivs \
       --outlier-mask \
       --camera $camera \
       --expnum $expnum \
       --out-dir $outdir \
       --threads $ncores \
       >> $logfile 2>&1
